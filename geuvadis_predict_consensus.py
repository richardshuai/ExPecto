# -*- coding: utf-8 -*-
import argparse
import torch
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

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

ENFORMER_SEQ_LENGTH = 393216


def main():
    parser = argparse.ArgumentParser(description='Predict expression for consensus sequences using ExPecto')
    parser.add_argument('expecto_model')
    parser.add_argument('consensus_dir')
    parser.add_argument('genes_file')
    parser.add_argument('--beluga_model', type=str, default='./resources/deepsea.beluga.pth')
    parser.add_argument('--batch_size', action="store", dest="batch_size",
                        type=int, default=1024, help="Batch size for neural network predictions.")
    parser.add_argument('--overwrite', action="store_true", dest="overwrite", default=False, help="If true, overwrite existing predictions. Otherwise, skip if h5 file is present.")
    parser.add_argument('--exp_only', action="store_true", dest="exp_only", default=False, help="If true, load in chromatin preds and make predictions.")
    parser.add_argument("--num_chunks", action="store", dest="num_chunks", type=int, default=None, help="Total number of chunks to split predictions")
    parser.add_argument("--chunk_i", action="store", dest="chunk_i", type=int, default=None, help="Chunk index for current run, starting from 0")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_predict_consensus',
                        help='Output directory')
    args = parser.parse_args()

    consensus_dir = args.consensus_dir
    genes_file = args.genes_file

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

    # get strand
    genes_df = pd.read_csv(genes_file, names=['ens_id', 'chrom', 'bp', 'gene_symbol', 'strand'], index_col=False)
    genes_df['gene_symbol'] = genes_df['gene_symbol'].fillna(genes_df['ens_id']).str.lower()
    genes_df = genes_df.set_index('gene_symbol')

    # Split into chunks if options are set
    if args.num_chunks is not None:
        gene_splits = np.array_split(genes, args.num_chunks)
        genes = gene_splits[args.chunk_i]
        assert len(genes) > 0, "Gene split resulted in empty list"

    print("Predicting chromatin for all samples for all genes...")
    for gene in tqdm(genes):
        fasta_files = glob.glob(f'{consensus_dir}/{gene}/samples/*.fa')
        strand = genes_df.loc[gene, 'strand']

        preds_dir = f'{args.out_dir}/{gene}'
        os.makedirs(preds_dir, exist_ok=True)

        if not args.overwrite and os.path.exists(f'{preds_dir}/{gene}.h5'):
            # skip if output h5 file already exists
            print(f"Skipping gene {gene} since h5 is already present.")
            continue

        if args.exp_only:
            with h5py.File(f"{preds_dir}/{gene}_chromatin.h5", "r") as f:
                preds = np.array(f['chromatin_preds'])
                fasta_record_ids = [x.decode('utf-8') for x in f['record_ids']]
        else:
            fasta_record_ids = []
            sample_seqs_gen = gen_sample_seqs_and_id_for_gene(fasta_files)
            preds = []
            for sample_seq, record_id in sample_seqs_gen:
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

        # "backwards compatibility"
        features = np.sum(pos_weights[None, :, :, None] * preds[:, None, :, :], axis=2)
        features = np.concatenate([np.zeros((features.shape[0], 10, 1)), features], axis=2).reshape((-1, 20030))  # add 0 shift
        expecto_features = xgb.DMatrix(features)

        expecto_preds = bst.predict(expecto_features)

        with h5py.File(f'{preds_dir}/{gene}_chromatin.h5', 'w') as preds_h5:
            preds_h5.create_dataset('chromatin_preds', data=preds)  # n_samples x n_bins x n_features
            preds_h5.create_dataset('record_ids', data=np.array(fasta_record_ids, 'S'))

        with h5py.File(f'{preds_dir}/{gene}.h5', 'w') as preds_h5:
            preds_h5.create_dataset('expecto_preds', data=expecto_preds)
            preds_h5.create_dataset('record_ids', data=np.array(fasta_record_ids, 'S'))


def gen_sample_seqs_and_id_for_gene(fasta_files):
    """
    Create generator for 1-hot encoded sequences for input into Basenji2 for all samples for a given gene.
    fasta_gz: consensus seqs in the form of gzipped fasta e.g. {gene}/{gene}.fa.gz
    """
    for fasta_file in fasta_files:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            seq = str(record.seq).upper()

            interval = record.id.split(":")[1]
            if interval.startswith("-"):
                # if sequence has negative sign in interval, it means the sequence is definitely missing the beginning

                # sanity check
                bp_start = -int(interval.split("-")[-2])  # we need to parse like this because - sign is in front of first number
                bp_end = int(interval.split("-")[-1])
                assert bp_end - bp_start + 1 == ENFORMER_SEQ_LENGTH

                # pad with Ns to beginning of sequence
                seq = "N" * (ENFORMER_SEQ_LENGTH - len(seq)) + seq

            else:
                # sanity check
                bp_start, bp_end = map(int, interval.split("-"))
                assert bp_end - bp_start + 1 == ENFORMER_SEQ_LENGTH

                # check if sequence is missing end of sequence
                if len(seq) < ENFORMER_SEQ_LENGTH:
                    # pad with Ns to end of sequence
                    seq = seq + "N" * (ENFORMER_SEQ_LENGTH - len(seq))

            assert len(seq) == ENFORMER_SEQ_LENGTH, f"Sequence length is {len(seq)} for {record.id}"  # one last check
            yield seq, f"{record.id}|{Path(fasta_file).stem}"


def check_is_truncated(fasta_files: list) -> bool:
    """
    Check whether consensus sequences are truncated at the beginning or end of the gene.
    """
    fasta_file = fasta_files[0]  # check first file
    for record in SeqIO.parse(fasta_file, 'fasta'):
        seq = str(record.seq).upper()

        interval = record.id.split(":")[1]
        if interval.startswith("-"):
            # if sequence has negative sign in interval, it means the sequence is definitely missing the beginning
            return True
        else:
            bp_start, bp_end = map(int, interval.split("-"))
            assert bp_end - bp_start + 1 == ENFORMER_SEQ_LENGTH
            # check if sequence is missing end of sequence
            if len(seq) < ENFORMER_SEQ_LENGTH:
                # pad with Ns to end of sequence
                return True

    return False




# def gen_sample_seqs_and_id_for_gene(fasta_gz):
#     """
#     Create generator for 1-hot encoded sequences for input into Basenji2 for all samples for a given gene.
#     fasta_gz: consensus seqs in the form of gzipped fasta e.g. {gene}/{gene}.fa.gz
#     """
#     sample_seqs = []
#     with gzip.open(fasta_gz, 'rt') as f:
#         for record in SeqIO.parse(f, 'fasta'):
#             seq = str(record.seq).upper()
#             yield seq, record.id


def get_seq_shifts_for_sample_seq(sample_seq, strand, shifts, windowsize=2000):
    """
    Get shifts for sequence, centered at TSS.
    windowsize denotes input size for neural network, which is 2000 for default Beluga model.
    """
    # assumes TSS is at center of sequence, with less sequence upstream of seq is even length. <-- outdated
    # We are using basenji consensus seqs as opposed to enformer consensus seqs here <-- outdated
    # if strand == '+':
    #     strand = 1
    #     tss_i = (len(sample_seq) - 1) // 2
    # elif strand == '-':
    #     strand = -1
    #     tss_i = len(sample_seq) // 2
    # else:
    #     assert False, f'strand {strand} not recognized'
    # Ok, now we are using enformer consensus seqs, so tss is always at len(sample_seq) // 2

    tss_i = len(sample_seq) // 2
    if strand == '+':
        strand = 1
    elif strand == '-':
        strand = -1
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
