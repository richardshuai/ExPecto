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
from expecto_utils import encodeSeqs
import shutil

def main():
    parser = argparse.ArgumentParser(description='Predict expression for consensus sequences using ExPecto')
    parser.add_argument('expecto_model')
    parser.add_argument('consensus_dir')
    parser.add_argument('eqtls_df_file')
    parser.add_argument('snps_vcf')
    parser.add_argument('--beluga_model', type=str, default='./resources/deepsea.beluga.pth')
    parser.add_argument('--batch_size', action="store", dest="batch_size",
                        type=int, default=1024, help="Batch size for neural network predictions.")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_sed_for_top_eqtls',
                        help='Output directory')
    args = parser.parse_args()

    expecto_model = args.expecto_model
    consensus_dir = args.consensus_dir
    eqtls_df_file = args.eqtls_df_file
    snps_vcf = args.snps_vcf

    # preserve vcf file
    shutil.copy(snps_vcf, f'{args.out_dir}/snps.vcf')

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Beluga()
    model.load_state_dict(torch.load(args.beluga_model))
    model.eval().to(device)

    bst = xgb.Booster()
    bst.load_model(args.expecto_model.strip())

    # load in eqtls file
    eur_top_eqtl_genes_df = pd.read_csv(eqtls_df_file)
    eur_top_eqtl_genes_df['gene_symbol'] = eur_top_eqtl_genes_df['name'].fillna(eur_top_eqtl_genes_df['geneID'])

    # preprocessing for join
    eur_top_eqtl_genes_df['SNPpos'] = eur_top_eqtl_genes_df['SNPpos'].astype(int).astype(str)
    eur_top_eqtl_genes_df = eur_top_eqtl_genes_df.set_index('chr' + eur_top_eqtl_genes_df['CHR_SNP'].astype(str) + '_' + eur_top_eqtl_genes_df['SNPpos'])

    # read vcf
    vcf_df = pd.read_csv(snps_vcf, sep='\t', comment='#', header=None).iloc[:, 0:5]
    vcf_df.columns = ['SNP_CHROM', 'SNP_POS', 'ID', 'REF', 'ALT']
    vcf_df.index = vcf_df.iloc[:, 0] + '_' + vcf_df.iloc[:, 1].astype(str)
    vcf_df = vcf_df.drop_duplicates()   # drop duplicated SNPs since they were taken from top eQTLs

    eqtls_df = eur_top_eqtl_genes_df.merge(vcf_df, left_index=True, right_index=True, validate='m:1', how='inner')  # why are we losing SNPs here as opposed to a left join?
    num_eqtls = eqtls_df.shape[0]

    # evaluate
    shifts = np.array(list(range(-20000, 20000, 200)))

    record_ids = []
    genes = []
    beluga_ref_preds = []
    beluga_alt_preds = []
    seqs_gen = seqs_to_predict(eqtls_df, consensus_dir)

    for i in tqdm(range(num_eqtls)):
        # Predict on reference
        ref_id, ref_seq = next(seqs_gen)
        record_ids.append(ref_id)
        genes.append(eqtls_df.iloc[i].loc['gene_symbol'])

        strand = eqtls_df.iloc[i].loc['strand']
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

    expecto_ref_features = xgb.DMatrix(
        np.sum(pos_weights[None, :, :, None] * beluga_ref_preds[:, None, :, :], axis=2).reshape(-1, 10 * 2002)
    )

    expecto_alt_features = xgb.DMatrix(
        np.sum(pos_weights[None, :, :, None] * beluga_alt_preds[:, None, :, :], axis=2).reshape(-1, 10 * 2002)
    )
    expecto_ref_preds = bst.predict(expecto_ref_features)
    expecto_alt_preds = bst.predict(expecto_alt_features)

    with h5py.File(f'{args.out_dir}/preds.h5', 'w') as preds_h5:
        preds_h5.create_dataset('ref_preds', data=expecto_ref_preds)
        preds_h5.create_dataset('alt_preds', data=expecto_alt_preds)
        preds_h5.create_dataset('record_ids', data=np.array(record_ids, 'S'))
        preds_h5.create_dataset('genes', data=np.array(genes, 'S'))


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


def seqs_to_predict(eqtls_df, consensus_dir):
    for _, eqtl in eqtls_df.iterrows():
        gene = eqtl['gene_symbol'].lower()
        ref_fasta = f'{consensus_dir}/{gene}/ref_roi.fa'
        ref_id, ref_seq = get_1_id_and_seq_from_fasta(ref_fasta)
        yield ref_id, ref_seq

        ref_allele, alt_allele = eqtl['REF'], eqtl['ALT']

        ref_chr = ref_id.split('|')[0].split(':')[0]
        ref_start, ref_end = map(int, ref_id.split('|')[0].split(':')[1].split('-'))
        seq_length = len(ref_seq)
        assert (ref_end - ref_start + 1) == seq_length, "record ID does not match fasta seq length"

        assert f'{eqtl["SNP_CHROM"]}' == ref_chr, "Chromosomes do not match between eQTL df and ref fasta id"
        if eqtl['strand'] == '+':
            assert eqtl['TSSpos'] == (ref_end - (seq_length // 2)), "TSSpos in eQTL file not consistent with fasta record"
        else:
            assert eqtl['TSSpos'] == (ref_start + (seq_length // 2)), "TSSpos in eQTL file not consistent with fasta record"

        if eqtl['strand'] == '+':
            tss_i = seq_length // 2 - 1
        else:
            tss_i = seq_length // 2
        snp_i = int(tss_i - (eqtl['TSSpos'] - eqtl['SNP_POS']))

        assert ref_seq[snp_i] == ref_allele, "Ref sequence does not match ref allele"
        alt_seq = ref_seq[:snp_i] + alt_allele + ref_seq[snp_i + 1:]

        # print(f"Reference sequence surrounding SNP: {ref_seq[snp_i-5:snp_i]}*{ref_seq[snp_i]}*{ref_seq[snp_i + 1:snp_i+6]}")
        # print(f"Alternate sequence surrounding SNP: {alt_seq[snp_i - 5:snp_i]}*{alt_seq[snp_i]}*{alt_seq[snp_i + 1:snp_i + 6]}")
        yield alt_seq


if __name__ == '__main__':
    main()
