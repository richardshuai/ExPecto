import pandas as pd
import numpy as np
import argparse
import glob
from natsort import natsorted
import os

def main():
    cell_type = 'PT'
    parser = argparse.ArgumentParser(description='Make gene expression annotation file for kidney data')
    parser.add_argument('--gene_anno_file', dest='gene_anno_file', type=str, default='./resources/geneanno.csv')
    parser.add_argument('--counts_dir', dest='counts_dir', type=str,
                      default='resources/bulk_rna_seq')
    parser.add_argument('--rank_match_file', dest='rank_match_file', default=None,
                        help='If provided, force counts to match this file based on rank')
    parser.add_argument('-i', dest='i', type=int, default=0, help='Index of column of rank_match_file'
                                                                  ' to rank match to, 0-indexed')
    parser.add_argument('--out_dir', dest='out_dir', type=str,
                        default='resources/bulk_rna_seq/geneannos')
    args = parser.parse_args()

    geneanno = pd.read_csv(args.gene_anno_file, index_col=0)

    counts_files = natsorted(glob.glob(f'{args.counts_dir}/*.tsv'))

    for counts_file in counts_files:
        counts_df = pd.read_csv(counts_file, sep='\t', header=0)
        counts_df['ens_id'] = counts_df['gene_id'].str.split('.').str[0]
        counts_df = pd.DataFrame(counts_df.groupby('ens_id')['FPKM'].sum(), columns=['FPKM'])  # TODO:  Check this is valid to do

        df_merged = geneanno.merge(counts_df, how='left', left_index=True, right_on='ens_id', validate='1:1')
        df_out = df_merged.loc[:, 'FPKM']
        df_out.index = range(1, len(df_out.index) + 1)

        if args.rank_match_file is not None:
            geneanno_to_match = pd.read_csv(args.rank_match_file, index_col=0)
            print(f"Matching to {geneanno_to_match.columns[args.i]}...")
            nan_mask = np.any(df_out.isnull(), axis=1)
            col_to_match = geneanno_to_match.iloc[:, args.i].reset_index(drop=True)
            # Only use genes present in the kidney cell types for matching
            col_to_match = col_to_match[~nan_mask.reset_index(drop=True)]
            sorted_vals = np.sort(col_to_match)

            for j in range(df_out.shape[1]):
                x = df_out.iloc[:, j][~nan_mask].argsort().argsort()  # Double argsort to obtain rankings
                df_out.iloc[:, j][~nan_mask] = sorted_vals[x]
                df_out.iloc[:, j][nan_mask] = np.nan

        exp_filename = os.path.splitext(os.path.basename(counts_file))[0]
        df_out.to_csv(f'{args.out_dir}/geneanno.exp_{exp_filename}.csv')


if __name__ == '__main__':
    main()
