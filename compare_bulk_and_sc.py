import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import glob

def main():
    parser = argparse.ArgumentParser(description='Compare bulk rna vs. pbmc single cell')
    parser.add_argument('--bulk_exp_dir', dest='bulk_exp_file', type=str, default='data/bulk_rna_seq/geneannos')
    parser.add_argument('--sc_exp_file', dest='sc_exp_file', type=str,
                        default='resources/geneanno.exp_pbmc.csv')
    parser.add_argument('--out_dir', type=str, default='bulk_sc_comparison')
    args = parser.parse_args()

    bulk_exp_files = glob.glob(f'{args.bulk_exp_dir}/geneanno.exp_*.csv')
    sc_counts_df = pd.read_csv(args.sc_exp_file, index_col=0).reset_index(drop=True)
    sc_counts_df.columns = ['CD4', 'CD8', 'CD14', 'B', 'NK']

    for col in sc_counts_df

if __name__ == '__main__':
    main()

