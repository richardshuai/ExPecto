import pandas as pd
import numpy as np
import argparse

def main():
    cell_type = 'PT'
    parser = argparse.ArgumentParser(description='Make gene expression annotation file for kidney data')
    parser.add_argument('--gene_anno_file', dest='gene_anno_file', type=str, default='./resources/geneanno.csv')
    parser.add_argument('--counts_file', dest='counts_file', type=str,
                      default='resources/GSM3196026_DS33394-LN29326-AG7008-genes.tsv')
    parser.add_argument('--rank_match_file', dest='rank_match_file', default=None,
                        help='If provided, force counts to match this file based on rank')
    parser.add_argument('-i', dest='i', type=int, default=0, help='Index of column of rank_match_file'
                                                                  ' to rank match to, 0-indexed')
    parser.add_argument('--out_file', dest='out_file', type=str,
                        default='resources/geneanno.exp_GSM3196026.csv')
    args = parser.parse_args()

    geneanno = pd.read_csv(args.gene_anno_file, index_col=0)
    counts_df = pd.read_csv(args.counts_file, sep='\t', header=0, index_col=0)
    counts_df['cultured_PT'] = np.exp2(counts_df['logrpm']) - 0.1  # from log2(RPM + 0.1) back to RPM
    counts_df = counts_df['cultured_PT']

    # df_merged = geneanno.merge(counts_df, how='left', left_on='symbol', right_index=True, validate='m:1')
    # TODO: Deal with the duplicate gene symbols
    # duplicated = geneanno[geneanno["symbol"].duplicated(keep="first")]
    # print(f'Number of duplicated genes: {len(duplicated)}')

    # Using Ensembl IDs -- no duplicates
    df_merged = geneanno.merge(counts_df, how='left', left_index=True, right_index=True, validate='1:1')

    df_out = df_merged.loc[:, ['cultured_PT']]
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

    df_out.to_csv(args.out_file)


if __name__ == '__main__':
    main()
