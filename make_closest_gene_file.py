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
    parser.add_argument('--geneanno_file', dest='geneanno_file', type=str, default='./resources/geneanno.csv')
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_closest_gene_file',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vcf = pd.read_csv(args.hg19_snps_file, sep='\t', header=None, comment='#')
    geneanno = pd.read_csv(args.geneanno_file, index_col=0)

    closest_gene_df = pd.DataFrame(columns=('snp_chrom', 'snp_pos_start', 'snp_pos', 'ref', 'alt',
                                            'tss_chrom', 'tss_pos_start', 'tss_pos', 'tss_strand', 'ens_id',
                                            'dist_to_tss'))

    for i, row in vcf.iterrows():
        snp_chrom, snp_pos, ref, alt = row[0], row[1], row[3], row[4]

        # Get closest gene to SNP
        # TODO: Can we get multiple genes in the receptive field instead?
        gene_row = find_closest_gene(snp_chrom, snp_pos, geneanno)
        tss_chrom, tss_pos, tss_strand = gene_row['seqnames'], gene_row['CAGE_representative_TSS'], gene_row['strand']
        ens_id = gene_row.name
        dist_to_tss = tss_pos - snp_pos

        closest_gene_df.loc[i] = [snp_chrom[3:], snp_pos - 1, snp_pos, ref, alt, tss_chrom[3:], tss_pos - 1, tss_pos,
                                     tss_strand, ens_id, dist_to_tss]

    closest_gene_df.to_csv(f'{args.out_dir}/closest_gene.tsv', sep='\t', index=False, header=False)


def find_closest_gene(snp_chrom, snp_pos, geneanno):
    """
    Returns row in gene anno of closest gene to snp_chrom, snp_pos.
    - snp_chrom: chromosome of SNP, e.g. "chr1"
    - snp_pos: position of SNP
    """
    geneanno = geneanno.loc[geneanno['seqnames'] == snp_chrom]
    geneanno['dists'] = geneanno['CAGE_representative_TSS'] - snp_pos
    closest_i = np.argmin(np.abs(geneanno['dists']))
    return geneanno.iloc[closest_i]


if __name__ == '__main__':
    main()
