# -*- coding: utf-8 -*-
import argparse
import glob
import os

import h5py
import numpy as np
import torch
import xgboost as xgb
from Bio import SeqIO
from natsort import natsorted
from tqdm import tqdm

from Beluga import Beluga
from expecto_utils import encodeSeqs
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Predict expression for consensus sequences using ExPecto')
    parser.add_argument('expecto_model')
    parser.add_argument('consensus_dir')
    parser.add_argument('genes_file')
    parser.add_argument('--beluga_model', type=str, default='./resources/deepsea.beluga.pth')
    parser.add_argument('--batch_size', action="store", dest="batch_size",
                        type=int, default=1024, help="Batch size for neural network predictions.")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_sed_for_top_eqtls',
                        help='Output directory')
    args = parser.parse_args()

    expecto_model = args.expecto_model
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

    genes_df = pd.read_csv(genes_file, names=['ens_id', 'chrom', 'bp', 'gene_symbol', 'strand'], index_col=False)
    genes_df['gene_symbol'] = genes_df['gene_symbol'].fillna(genes_df['ens_id'])
    genes_df = genes_df.set_index('gene_symbol')

    record_ids = []
    expecto_ref_preds = []
    for i, gene in enumerate(tqdm(genes_df.index)):
        strand = genes_df.loc[gene, 'strand']

        # Predict on reference
        gene = gene.lower()
        ref_fasta = f'{consensus_dir}/{gene}/ref_roi.fa'
        ref_id, ref_seq = get_1_id_and_seq_from_fasta(ref_fasta)

        if strand == '-':
            # TODO: needed due to ref seq glitch not properly reverse complemented
            ref_seq = reverse_complement(ref_seq.upper())

        record_ids.append(ref_id)
        seq_shifts = encodeSeqs(get_seq_shifts_for_sample_seq(ref_seq, strand, shifts)).astype(np.float32)
        ref_preds = np.zeros((seq_shifts.shape[0], 2002))
        for i in range(0, seq_shifts.shape[0], args.batch_size):
            batch = torch.from_numpy(seq_shifts[i * args.batch_size:(i + 1) * args.batch_size]).to(device)
            batch = batch.unsqueeze(2)
            ref_preds[i * args.batch_size:(i + 1) * args.batch_size] = model.forward(batch).cpu().detach().numpy()

        # avg the reverse complement
        ref_preds = (ref_preds[:ref_preds.shape[0] // 2] + ref_preds[ref_preds.shape[0] // 2:]) / 2
        beluga_ref_preds = np.array(ref_preds)[None]

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

        expecto_ref_features = xgb.DMatrix(
            np.sum(pos_weights[None, :, :, None] * beluga_ref_preds[:, None, :, :], axis=2).reshape(-1, 10 * 2002)
        )

        expecto_ref_preds.append(bst.predict(expecto_ref_features))

    expecto_ref_preds = np.array(expecto_ref_preds).squeeze()

    # compute log transform
    expecto_ref_preds = np.exp(expecto_ref_preds) - 0.0001  # invert log transform from ExPecto
    expecto_ref_preds = np.log10(expecto_ref_preds + 0.1)

    df = pd.DataFrame({"genes": np.array(genes_df.index.values), "ref_preds": expecto_ref_preds})
    df.to_csv(f'{args.out_dir}/ref_preds.csv', header=True, index=False)

    # with h5py.File(f'{args.out_dir}/preds.h5', 'w') as preds_h5:
    #     preds_h5.create_dataset('ref_preds', data=expecto_ref_preds)
    #     preds_h5.create_dataset('record_ids', data=np.array(record_ids, 'S'))
    #     preds_h5.create_dataset('genes', data=np.array(genes_df.index.values, 'S'))


def get_1_id_and_seq_from_fasta(fasta_file):
    """
    Get sequence from fasta file. Fasta file should only have 1 record. Automatically upper cases all nts.
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    assert len(records) == 1, f"Expected 1 record in fasta file {fasta_file}, but got {len(records)} records"
    return records[0].id, str(records[0].seq).upper()


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


def reverse_complement(x):
    complements = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(list(map(complements.get, list(x)))[::-1])


if __name__ == '__main__':
    main()
