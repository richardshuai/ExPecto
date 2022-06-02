import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import h5py

def main():
    parser = argparse.ArgumentParser(description='Plot metrics from trained Susztak models')
    parser.add_argument('--metrics_h5', type=str, default='metrics h5')
    parser.add_argument('--out_dir', type=str, default='temp_plot_susztak')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    metrics_h5 = h5py.File(args.metrics_h5, 'r')

    def scatter_hist(x, y, ax, ax_histx, ax_histy, xlabel, ylabel):
        # no labels
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # the scatter plot:
        # now determine nice limits by hand:
        xymax = max(np.max(x), np.max(y))
        xymin = min(np.min(x), np.min(y))
        max_lim = xymax + 0.002
        min_lim = xymin - 0.002

        ax.scatter(x[:-1], y[:-1], c='black', s=30)
        ax.scatter(x[-1:], y[-1:], c='orange', s=30, label='averaged expression')
        ax.legend()
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        binwidth = (xymax - xymin) / 15


        bins = np.arange(xymin, xymax, binwidth)
        ax_histx.hist(x, bins=bins, alpha=0.8)
        ax_histy.hist(y, bins=bins, orientation='horizontal', alpha=0.8)

    # definitions for the axes
    left, width = 0.12, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(metrics_h5['pearsonr_trains'], metrics_h5['pearsonr_valids'], ax, ax_histx, ax_histy,
                 xlabel="Train PearsonR (holding out chr8, chr7)",
                 ylabel="Valid PearsonR (chr8)")
    plt.tight_layout()
    plt.savefig(f'{args.out_dir}/pearsonr.png', dpi=300)
    plt.show()

    # start with a square Figure
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(metrics_h5['r2_trains'], metrics_h5['r2_valids'], ax, ax_histx, ax_histy,
                 xlabel="Train r2 (holding out chr8, chr7)",
                 ylabel="Valid r2 (chr8)")
    plt.tight_layout()
    plt.savefig(f'{args.out_dir}/r2.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    main()

