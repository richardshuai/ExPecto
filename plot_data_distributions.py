import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

def main():
    parser = argparse.ArgumentParser(description='Make gene expression annotation file for kidney data')
    parser.add_argument('--exp_file', dest='exp_file', type=str, default='./resources/geneanno.exp.csv')
    parser.add_argument('--kidney_exp_file', dest='kidney_exp_file', type=str,
                        default='resources/geneanno.exp_kidney.csv')
    parser.add_argument('--pseudocount', action="store",
                        dest="pseudocount", type=float, default=0.0001)
    parser.add_argument('--out_dir', type=str, default='data_distribution_plots')
    parser.add_argument('--kidney_genes_only', action="store_true",
                        dest="kidney_genes_only", default=False,
                        help="If true, only use genes in our kidney data.")
    args = parser.parse_args()

    # Plot kidney data distributions
    plot_dir = f'{args.out_dir}/kidney'
    os.makedirs(plot_dir, exist_ok=True)
    kidney_exp_df = pd.read_csv(args.kidney_exp_file, index_col=0).reset_index(drop=True)
    nan_mask = np.any(kidney_exp_df.isnull(), axis=1)
    kidney_exp_df = kidney_exp_df[~nan_mask]
    kidney_exp_df = np.log(kidney_exp_df + args.pseudocount)
    xmin, xmax = np.min(np.array(kidney_exp_df)), np.max(np.array(kidney_exp_df))
    bins = np.linspace(xmin, xmax, num=50)
    for i, cell_type in enumerate(kidney_exp_df):
        # if i == 3:
        #     break
        plt.figure()
        plt.hist(kidney_exp_df.loc[:, cell_type], bins=bins)
        plt.title(f'{cell_type}')
        plt.savefig(f'{plot_dir}/{cell_type}_hist.png', dpi=300)
        plt.show()

    # Plot ExPecto data distribution
    plot_dir = f'{args.out_dir}/expecto'
    os.makedirs(plot_dir, exist_ok=True)
    exp_df = pd.read_csv(args.exp_file, index_col=0).reset_index(drop=True)
    exp_df = np.log(exp_df + args.pseudocount)
    if args.kidney_genes_only:
        print("Only using genes found in our kidney data")
        exp_df = exp_df[~nan_mask]
    xmin, xmax = np.min(np.array(exp_df)), np.max(np.array(exp_df))
    bins = np.linspace(xmin, xmax, num=50)
    for i, cell_type in enumerate(exp_df):
        if 'kidney' not in cell_type.lower():
            continue
        plt.figure()
        plt.hist(exp_df.loc[:, cell_type], bins=bins)
        plt.title(f'{cell_type}')
        plt.savefig(f'{plot_dir}/{cell_type}_hist.png', dpi=300)
        plt.show()

    # Plot data distributions for PT against ExPecto
    kidney_cell_type = 'PT'
    i_e = 0
    expecto_cell_type = exp_df.columns[i_e]

    if args.kidney_genes_only:
        # Mask is already applied to exp_df
        plot_kidney_vs_expecto(x_kidney=kidney_exp_df[kidney_cell_type], y_expecto=exp_df[expecto_cell_type],
                               xlabel=f'{kidney_cell_type} expression, log(RPKM)',
                               ylabel=f'{expecto_cell_type} expression, log(RPKM)',
                               out_dir=f'{args.out_dir}/scatter_{kidney_cell_type}_vs_{expecto_cell_type}.png')

    else:
        plot_kidney_vs_expecto(x_kidney=kidney_exp_df[kidney_cell_type], y_expecto=exp_df[~nan_mask][expecto_cell_type],
                               xlabel=f'{kidney_cell_type} expression, log(RPKM)',
                               ylabel=f'{expecto_cell_type} expression, log(RPKM)',
                               out_dir=f'{args.out_dir}/scatter_{kidney_cell_type}_vs_{expecto_cell_type}.png')


# Plots
def plot_kidney_vs_expecto(x_kidney, y_expecto, xlabel, ylabel, out_dir):
    spearman, _ = spearmanr(x_kidney, y_expecto)
    pearson, _ = pearsonr(x_kidney, y_expecto)
    fig = sns.scatterplot(x=x_kidney, y=y_expecto, color="black", alpha=0.3, s=10)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(np.min([y_expecto, x_kidney]), np.max([y_expecto, x_kidney]))
    plt.ylim(np.min([y_expecto, x_kidney]), np.max([y_expecto, x_kidney]))
    plt.title(f'PearsonR: {pearson:.3f}, SpearmanR: {spearman:.3f}')
    plt.savefig(out_dir, dpi=300)
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()

