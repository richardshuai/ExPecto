# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
from tqdm import tqdm
import os

from Beluga import Beluga
from expecto_utils import encodeSeqs
from natsort import natsorted
import glob
import gzip
from Bio import SeqIO
import h5py
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def main():
    parser = argparse.ArgumentParser(description='Predict expression for consensus sequences using ExPecto')
    parser.add_argument('expecto_model')
    parser.add_argument('consensus_dir')
    parser.add_argument('--beluga_model', type=str, default='./resources/deepsea.beluga.pth')
    parser.add_argument('--batch_size', action="store", dest="batch_size",
                        type=int, default=1024, help="Batch size for neural network predictions.")
    parser.add_argument('--eighth_split', action='store', dest='eighth_split',
                        type=int, default=None, help='Split into eighths for computation. one of [0, 1, 2, 3, 4, 5, 6, 7 or None if all.')
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_predict_consensus',
                        help='Output directory')
    args = parser.parse_args()

    expecto_model = args.expecto_model
    consensus_dir = args.consensus_dir

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Beluga()
    model.load_state_dict(torch.load(args.beluga_model))
    model.eval().to(device)

    bst = xgb.Booster()
    bst.load_model(args.expecto_model.strip())

    # evaluate
    shifts = np.array(list(range(-20000, 20000, 200)))
    genes = natsorted([os.path.basename(file) for file in glob.glob(f'{consensus_dir}/*')])

    # Split into eighths if option is set
    if args.eighth_split is None:
        gene_splits = np.array_split(genes, 8)
        genes = gene_splits[args.eighth_split]
        assert len(genes) > 0, "Gene split resulted in empty list"

    print("Predicting chromatin for all samples for all genes...")
    for gene in tqdm(genes):
        fasta_gz = f'{consensus_dir}/{gene}/{gene}.fa.gz'

        preds_dir = f'{args.out_dir}/{gene}'
        os.makedirs(preds_dir, exist_ok=True)

        fasta_record_ids = []
        sample_seqs_gen = gen_sample_seqs_and_id_for_gene(fasta_gz)
        preds = []
        for sample_seq, record_id in tqdm(sample_seqs_gen):
            strand = record_id.split('|')[-2]
            seq_shifts = encodeSeqs(get_seq_shifts_for_sample_seq(sample_seq, strand, shifts)).astype(np.float32)

            sample_preds = np.zeros((seq_shifts.shape[0], 2002))
            for i in range(0, seq_shifts.shape[0], args.batch_size):
                batch = torch.from_numpy(seq_shifts[i * args.batch_size:(i+1) * args.batch_size]).to(device)
                batch = batch.unsqueeze(2)
                sample_preds[i * args.batch_size:(i+1) * args.batch_size] = model.forward(batch).cpu().detach().numpy()

            # avg the reverse complement
            sample_preds = (sample_preds[:sample_preds.shape[0] // 2] + sample_preds[sample_preds.shape[0] // 2:]) / 2
            fasta_record_ids.append(record_id)
            preds.append(sample_preds)

        preds = np.stack(preds, axis=0)
        # assert preds.shape == (len(fasta_record_ids), shifts.shape[0], 2002)

        pos_weight_shifts = shifts
        pos_weights = np.vstack([
            np.exp(-0.01 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.02 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.05 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.1 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.2 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts <= 0),
            np.exp(-0.01 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.02 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.05 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.1 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0),
            np.exp(-0.2 * np.abs(pos_weight_shifts) / 200) * (pos_weight_shifts >= 0)])

        expecto_features = xgb.DMatrix(
            np.sum(pos_weights[None, :, :, None] * preds[:, None, :, :], axis=2).reshape(-1, 10 * 2002)
        )

        expecto_preds = bst.predict(expecto_features)


        # genes_file = '/home/rshuai/research/ni-lab/analysis/geuvadis_data/eur_eqtl_genes_converted.csv'
        # geuvadis_expression_file = '/home/rshuai/research/ni-lab/analysis/geuvadis_data/GD462.GeneQuantRPKM.50FN.samplename.resk10.txt'
        #
        # genes_df = pd.read_csv(genes_file, names=['ens_id', 'chrom', 'bp', 'gene_symbol', 'strand'], index_col=False)
        # genes_df['gene_symbol'] = genes_df['gene_symbol'].fillna(genes_df['ens_id'])
        # genes_df = genes_df.set_index('gene_symbol')

        # Read Geuvadis data
        # geuvadis_exp_df = pd.read_csv(geuvadis_expression_file, sep='\t')
        # geuvadis_exp_df['Gene_Symbol'] = geuvadis_exp_df['Gene_Symbol'].apply(lambda x: x.split('.')[0])  # Ensembl IDs
        # geuvadis_exp_df = geuvadis_exp_df.rename({'Gene_Symbol': 'ens_id'}, axis=1)
        #
        # preds_ti = expecto_preds
        # preds_df = pd.DataFrame({'record_ids': fasta_record_ids, 'preds': preds_ti})
        # records_df = pd.DataFrame(preds_df['record_ids'].str.split('|').values.tolist(),
        #                           columns=['chrom:pos', 'sample', 'strand', 'haplotype'])
        # preds_df['sample'] = records_df['sample']
        # preds_df = preds_df.groupby('sample')['preds'].mean().reset_index()
        # gene_exp = geuvadis_exp_df[geuvadis_exp_df['ens_id'] == genes_df.loc[gene.upper()]['ens_id']].T.iloc[4:].astype(float)
        # preds_df = preds_df.merge(gene_exp, how='inner', left_on='sample', right_index=True, validate='1:1')
        # plot_preds(np.log(preds_df.iloc[:, -1]).values, preds_df['preds'].values,
        #            title='ExPecto Predictions vs. Geuvadis',
        #            xlabel='log(Geuvadis)',
        #            ylabel='log(prediction)',
        #            out_dir=f'{args.out_dir}/expecto_pred_vs_geuvadis.png')

        with h5py.File(f'{preds_dir}/{gene}.h5', 'w') as preds_h5:
            preds_h5.create_dataset('preds', data=expecto_preds)
            preds_h5.create_dataset('record_ids', data=np.array(fasta_record_ids, 'S'))


def gen_sample_seqs_and_id_for_gene(fasta_gz):
    """
    Create generator for 1-hot encoded sequences for input into Basenji2 for all samples for a given gene.
    fasta_gz: consensus seqs in the form of gzipped fasta e.g. {gene}/{gene}.fa.gz
    """
    sample_seqs = []
    with gzip.open(fasta_gz, 'rt') as f:
        for record in SeqIO.parse(f, 'fasta'):
            seq = str(record.seq).upper()
            yield seq, record.id


def get_seq_shifts_for_sample_seq(sample_seq, strand, shifts, windowsize=2000):
    """
    Get shifts for sequence, centered at TSS.
    windowsize denotes input size for neural network, which is 2000 for default Beluga model.
    """
    # assumes TSS is at center of sequence, with less sequence upstream of seq is even length
    if strand == '+':
        strand = 1
        tss_i = (len(sample_seq) - 1) // 2
    elif strand == '-':
        strand = -1
        tss_i = len(sample_seq) // 2
    else:
        assert False, f'strand {strand} not recognized'

    seq_shifts = []
    for shift in shifts:
        seq = list(sample_seq[tss_i + (shift * strand) - int(windowsize / 2 - 1):
                         tss_i + (shift * strand) + int(windowsize / 2) + 1])

        assert len(seq) == windowsize, f"Expected seq of length f{windowsize} but got {len(seq)}"
        seq_shifts.append(seq)

    return np.vstack(seq_shifts)


def plot_preds(ytrue, ypred, title, xlabel, ylabel, out_dir):
    print(spearmanr(ytrue, ypred))
    fig = sns.scatterplot(x=ytrue, y=ypred, color="black", alpha=0.3, s=20)
    plt.plot([0, 1], [0, 1], c='orange', transform=fig.transAxes)
    plt.xlim(np.min(ytrue), np.max(ytrue))
    plt.ylim(np.min(ypred), np.max(ypred))
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    pearson_r, _ = pearsonr(ytrue, ypred)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    spearman_r, _ = spearmanr(ytrue, ypred)
    plt.title(f'{title}\nPearsonR: {pearson_r:.3f}, R2: {r2:.3f}, SpearmanR: {spearman_r:.3f}')
    plt.savefig(out_dir, dpi=300)
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
