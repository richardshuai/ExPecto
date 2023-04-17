# -*- coding: utf-8 -*-
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from Bio import SeqIO
from tqdm import tqdm

from Beluga import Beluga
from pathlib import Path
from expecto_utils import encodeSeqs
import shutil

ENFORMER_SEQ_LENGTH = 393216


def main():
    parser = argparse.ArgumentParser(description='Predict expression for consensus sequences using ExPecto')
    parser.add_argument('expecto_model')
    parser.add_argument('consensus_dir')
    parser.add_argument('eur_top_eqtl_genes_csv')
    parser.add_argument('eqtls_csv')

    parser.add_argument('--beluga_model', type=str, default='./resources/deepsea.beluga.pth')
    parser.add_argument('--batch_size', action="store", dest="batch_size",
                        type=int, default=1024, help="Batch size for neural network predictions.")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_sed_for_top_eqtls',
                        help='Output directory')
    args = parser.parse_args()

    expecto_model = args.expecto_model
    consensus_dir = args.consensus_dir
    eur_top_eqtl_genes_csv = args.eur_top_eqtl_genes_csv
    eqtls_csv = args.eqtls_csv

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Beluga()
    model.load_state_dict(torch.load(args.beluga_model))
    model.eval().to(device)

    bst = xgb.Booster()
    bst.load_model(args.expecto_model.strip())

    # read in eqtls df and add strand to eqtls df
    eqtls_df = pd.read_csv(eqtls_csv)
    all_eqtls_df = pd.read_csv(eur_top_eqtl_genes_csv, names=["ens_id", "chr", "pos", "gene", "strand"])
    all_eqtls_df["gene"] = all_eqtls_df["gene"].str.lower()
    all_eqtls_df["gene"] = all_eqtls_df["gene"].fillna(all_eqtls_df["ens_id"].str.lower())
    eqtls_df["strand"] = pd.merge(eqtls_df, all_eqtls_df, left_on="name", right_on="gene", how="left")["strand"]

    num_eqtls = eqtls_df.shape[0]

    # evaluate
    shifts = np.array(list(range(-20000, 20000, 200)))

    genes = []
    beluga_ref_preds = []
    beluga_alt_preds = []
    seqs_gen = seqs_to_predict(eqtls_df, consensus_dir)

    for i in tqdm(range(num_eqtls)):
        strand = eqtls_df.iloc[i].loc['strand']

        # Predict on reference
        _, ref_seq = next(seqs_gen)
        genes.append(eqtls_df.iloc[i].loc['name'])

        seq_shifts = encodeSeqs(get_seq_shifts_for_sample_seq(ref_seq, strand, shifts)).astype(np.float32)
        ref_preds = np.zeros((seq_shifts.shape[0], 2002))
        for i in range(0, seq_shifts.shape[0], args.batch_size):
            batch = torch.from_numpy(seq_shifts[i * args.batch_size:(i + 1) * args.batch_size]).to(device)
            batch = batch.unsqueeze(2)
            ref_preds[i * args.batch_size:(i + 1) * args.batch_size] = model.forward(batch).cpu().detach().numpy()

        # avg the reverse complement
        ref_preds = (ref_preds[:ref_preds.shape[0] // 2] + ref_preds[ref_preds.shape[0] // 2:]) / 2
        beluga_ref_preds.append(ref_preds)

        # Predict on alternate
        alt_seq = next(seqs_gen)

        seq_shifts = encodeSeqs(get_seq_shifts_for_sample_seq(alt_seq, strand, shifts)).astype(np.float32)
        alt_preds = np.zeros((seq_shifts.shape[0], 2002))
        for i in range(0, seq_shifts.shape[0], args.batch_size):
            batch = torch.from_numpy(seq_shifts[i * args.batch_size:(i + 1) * args.batch_size]).to(device)
            batch = batch.unsqueeze(2)
            alt_preds[i * args.batch_size:(i + 1) * args.batch_size] = model.forward(batch).cpu().detach().numpy()

        # avg the reverse complement
        alt_preds = (alt_preds[:alt_preds.shape[0] // 2] + alt_preds[alt_preds.shape[0] // 2:]) / 2
        beluga_alt_preds.append(alt_preds)

    beluga_ref_preds = np.stack(beluga_ref_preds, axis=0)
    beluga_alt_preds = np.stack(beluga_alt_preds, axis=0)

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
    ref_features = np.sum(pos_weights[None, :, :, None] * beluga_ref_preds[:, None, :, :], axis=2).reshape(-1, 10 * 2002)
    ref_features = np.concatenate([np.zeros((ref_features.shape[0], 10, 1)), ref_features.reshape((-1, 10, 2002))], axis=2).reshape((-1, 20030))  # add 0 shift
    expecto_ref_features = xgb.DMatrix(ref_features)

    # "backwards compatibility"
    alt_features = np.sum(pos_weights[None, :, :, None] * beluga_alt_preds[:, None, :, :], axis=2).reshape(-1, 10 * 2002)
    alt_features = np.concatenate([np.zeros((alt_features.shape[0], 10, 1)), alt_features.reshape((-1, 10, 2002))], axis=2).reshape((-1, 20030))  # add 0 shift
    expecto_alt_features = xgb.DMatrix(alt_features)

    expecto_ref_preds = bst.predict(expecto_ref_features)
    expecto_alt_preds = bst.predict(expecto_alt_features)

    for i, gene in enumerate(genes):
        preds_dir = f'{args.out_dir}/{gene}'
        os.makedirs(preds_dir, exist_ok=True)

        with h5py.File(f'{preds_dir}/{gene}.h5', 'w') as preds_h5:
            preds_h5.create_dataset('ref_preds', data=expecto_ref_preds[i])
            preds_h5.create_dataset('alt_preds', data=expecto_alt_preds[i])


def get_1_id_and_seq_from_fasta(fasta_file):
    """
    Get sequence from fasta file. Fasta file should only have 1 record. Automatically upper cases all nts.
    Also deals with possible seq truncations from being at beginning or end of chromosome by padding appropriately with Ns.
    """
    records = list(SeqIO.parse(fasta_file, "fasta"))
    assert len(records) == 1, f"Expected 1 record in fasta file {fasta_file}, but got {len(records)} records"

    record = records[0]
    seq = str(record.seq).upper()

    # deal with possible seq truncations from being at beginning or end of chromosome
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
    return f"{record.id}|{Path(fasta_file).stem}", seq


def get_seq_shifts_for_sample_seq(sample_seq, strand, shifts, windowsize=2000):
    """
    Get shifts for sequence, centered at TSS.
    windowsize denotes input size for neural network, which is 2000 for default Beluga model.
    """
    # assumes that for the input sequence, the TSS is at position len(sample_seq) // 2 regardless of strand
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


def seqs_to_predict(eqtls_df, consensus_dir):
    for _, eqtl in eqtls_df.iterrows():
        gene = eqtl['name'].lower()
        strand = eqtl['strand']
        ref_fasta = f'{consensus_dir}/{gene}/ref.fa'

        # get fasta seq and validate against eqtl df
        ref_id, ref_seq = get_1_id_and_seq_from_fasta(ref_fasta)

        fasta_seq_length = len(ref_seq)
        ref_id = ref_id.split("|")[0]
        ref_chr = int(ref_id.split(':')[0].replace("chr", ""))
        ref_start, ref_end = map(int, ref_id.split(':')[1].split('-'))
        assert (ref_end - ref_start + 1) == fasta_seq_length, "record ID does not match fasta seq length"

        ref_allele, alt_allele = eqtl['REF'], eqtl['ALT']

        assert eqtl["CHR_SNP"] == ref_chr, "Chromosomes do not match between eQTL df and ref fasta id"
        assert eqtl['TSSpos_x'] == (ref_start + (fasta_seq_length // 2)), "TSSpos in eQTL file not consistent with fasta record"

        yield ref_id, ref_seq

        tss_i = len(ref_seq) // 2  # to input into seq_shifts_for_sample_seq, tss_i should be at index len(ref_seq) // 2
        snp_i = int(tss_i - (eqtl['TSSpos_x'] - eqtl['SNPpos']))

        assert ref_seq[snp_i] == ref_allele, "Ref sequence does not match ref allele"

        # print(f"Reference sequence surrounding SNP: {ref_seq[snp_i-5:snp_i]}*{ref_seq[snp_i]}*{ref_seq[snp_i + 1:snp_i+6]}")
        # print(f"Reference allele: {ref_allele}")
        # print(f"Strand of reference sequence: {strand}")

        alt_seq = ref_seq[:snp_i] + alt_allele + ref_seq[snp_i + 1:]
        # print(f"Alternate sequence surrounding SNP: {alt_seq[snp_i - 5:snp_i]}*{alt_seq[snp_i]}*{alt_seq[snp_i + 1:snp_i + 6]}")

        yield alt_seq


if __name__ == '__main__':
    main()
