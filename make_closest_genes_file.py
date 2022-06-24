# -*- coding: utf-8 -*-
import argparse
import math
import pyfasta
import torch
from torch import nn
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import os
from liftover import get_lifter

# Script based on https://github.com/FunctionLab/ExPecto/issues/9


def main():
    parser = argparse.ArgumentParser(description='Make closest gene file required by predict.py')
    parser.add_argument('hg19_snps_file')
    parser.add_argument('--all_in_receptive_field', action='store_true')
    parser.add_argument('--add_chr_prefix', action='store_true')
    parser.add_argument('--geneanno_file', dest='geneanno_file', type=str, default='./resources/geneanno.csv')
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_closest_gene_file',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vcf = pd.read_csv(args.hg19_snps_file, sep='\t', header=None, comment='#')
    if args.add_chr_prefix:
        vcf[0] = 'chr' + vcf[0].astype(str)  # include chr to chrom numbering in vcf
    geneanno = pd.read_csv(args.geneanno_file, index_col=0)

    # Create VCF file containing SNPs with multiplicity equal to number of genes in receptive field
    # TODO: This is not used in predict.py. Use the VCF from chromatin.py instead.
    vcf_out_path = f'{args.out_dir}/snps_hg19.vcf'
    vcf_file_out = open(vcf_out_path, 'w')
    print('##fileformat=VCFv4.3', file=vcf_file_out)
    print('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO', file=vcf_file_out)
    vcf_file_out.close()

    vcf_out_df = pd.DataFrame(columns=np.arange(vcf.shape[1]))
    closest_gene_df = pd.DataFrame(columns=('snp_chrom', 'snp_pos_start', 'snp_pos', 'ref', 'alt',
                                            'tss_chrom', 'tss_pos_start', 'tss_pos', 'tss_strand', 'ens_id',
                                            'dist_to_tss'))
    idx = 0
    for _, row in tqdm(vcf.iterrows(), total=vcf.shape[0]):
        snp_chrom, snp_pos, ref, alt = row[0], row[1], row[3], row[4]

        # Get closest gene to SNP
        if args.all_in_receptive_field:
            genes_df = get_genes_in_receptive_field(snp_chrom, snp_pos, geneanno)
        else:
            genes_df = find_closest_gene(snp_chrom, snp_pos, geneanno)

        for ens_id, gene_row in genes_df.iterrows():
            tss_chrom, tss_pos, tss_strand = gene_row['seqnames'], gene_row['CAGE_representative_TSS'], gene_row['strand']
            dist_to_tss = tss_pos - snp_pos

            vcf_out_df.loc[idx] = row
            closest_gene_df.loc[idx] = [snp_chrom[3:], snp_pos - 1, snp_pos, ref, alt, tss_chrom[3:], tss_pos - 1,
                                          tss_pos, tss_strand, ens_id, dist_to_tss]
            idx += 1


    closest_gene_df.to_csv(f'{args.out_dir}/closest_genes.tsv', sep='\t', index=False, header=False)
    vcf_out_df.to_csv(vcf_out_path, sep='\t', header=False, index=False, mode='a')


def find_closest_gene(snp_chrom, snp_pos, geneanno):
    """
    Returns row in gene anno of closest gene to snp_chrom, snp_pos.
    - snp_chrom: chromosome of SNP, e.g. "chr1"
    - snp_pos: position of SNP
    """
    geneanno = geneanno.loc[geneanno['seqnames'] == snp_chrom]
    geneanno['dists'] = geneanno['CAGE_representative_TSS'] - snp_pos
    closest_i = np.argmin(np.abs(geneanno['dists']).values)
    return geneanno.iloc[closest_i:closest_i + 1]


def get_genes_in_receptive_field(snp_chrom, snp_pos, geneanno):
    """
    Returns rows in gene anno of all genes within receptive field from snp_chrom, snp_pos, and tss strand.
    Receptive field is hardcoded assuming 20kb shifts and 2000bp input size.
    - snp_chrom: chromosome of SNP, e.g. "chr1"
    - snp_pos: position of SNP
    """
    geneanno = geneanno.loc[geneanno['seqnames'] == snp_chrom]
    geneanno['dists'] = geneanno['CAGE_representative_TSS'] - snp_pos

    # Filter by whether gene is in receptive field
    shifts = np.array(list(range(-20000, 20000, 200)))
    windowsize = 1000

    geneanno_rf = geneanno[geneanno.apply(lambda x: is_in_receptive_field(x, shifts, windowsize), axis=1)]
    if geneanno_rf.empty:
        closest_i = np.argmin(np.abs(geneanno['dists']).values)
        geneanno_rf = geneanno.iloc[closest_i:closest_i + 1]

    return geneanno_rf


def is_in_receptive_field(gene_row, shifts, windowsize):
    strand = 1 if gene_row.loc['strand'] == '+' else -1
    start = np.min((shifts * strand) - int(windowsize / 2 - 1))
    stop = np.max((shifts * strand) + int(windowsize / 2))

    return start <= -gene_row['dists'] <= stop


if __name__ == '__main__':
    main()
