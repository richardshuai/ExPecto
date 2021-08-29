import pandas as pd
import argparse

def main():
    cell_type = 'PT'
    parser = argparse.ArgumentParser(description='Make gene expression annotation file for kidney data')
    parser.add_argument('--gene_anno_file', dest='gene_anno_file', type=str, default='./resources/geneanno.csv')
    parser.add_argument('--counts_file', dest='counts_file', type=str,
                      default='/home/rshuai/research/ni-lab/analysis/basenji2/pseudobulk_rna/Wilson_rawcounts.txt')
    parser.add_argument('--out_file', dest='out_file', type=str,
                        default='resources/geneanno.exp_kidney.csv')
    args = parser.parse_args()

    geneanno = pd.read_csv(args.gene_anno_file, index_col=0)
    counts_df = pd.read_csv(args.counts_file, sep='\t', index_col=0)

    df_merged = geneanno.merge(counts_df, how='left', left_index=True, right_index=True)
    df_out = df_merged.iloc[:, -10:]
    df_out.index = range(1, len(df_out.index) + 1)
    df_out.to_csv(args.out_file)


if __name__ == '__main__':
    main()
