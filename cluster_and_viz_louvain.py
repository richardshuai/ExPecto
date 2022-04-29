# -*- coding: utf-8 -*-
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Orange.clustering import Louvain
from Orange.data import Table
from sklearn.manifold import TSNE

from cluster_utils import get_keep_mask


def main():
    parser = argparse.ArgumentParser(description='Cluster and visualize SVD-transformed data')
    parser.add_argument('svd_transform_dir')
    parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                        help="tsv file denoting Beluga features")
    parser.add_argument('--no_tf_features', action='store_true',
                        dest='no_tf_features', default=False,
                        help='leave out TF marks for training')
    parser.add_argument('--no_dnase_features', action='store_true',
                        dest='no_dnase_features', default=False,
                        help='leave out DNase marks for training')
    parser.add_argument('--no_histone_features', action='store_true',
                        dest='no_histone_features', default=False,
                        help='leave out histone marks for training')
    parser.add_argument('--intersect_with_lambert', action='store_true',
                        dest='intersect_with_lambert', default=False,
                        help='intersect with Lambert2018_TFs_v_1.01_curatedTFs.csv')
    parser.add_argument('--no_pol2', action='store_true',
                        dest='no_pol2', default=False,
                        help='take out Pol2*')
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_cluster_and_viz',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # setup
    np.random.seed(0)

    # read in npy files
    X = np.load(f'{args.svd_transform_dir}/tf_idf_reduced_100.npy')

    # dimensionality of SVD to use
    n_pcs = 20
    X = X[:, :n_pcs]

    # t-SNE
    louvain = Louvain(5)
    labels = louvain(Table(X))

    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X)

    plt.figure()
    for i in np.unique(labels):
        X_plot = X_embedded[labels == i]
        plt.scatter(X_plot[:, 0], X_plot[:, 1], label=f'cluster {i}')
    plt.show()

    # Save clustering
    beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
    beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df[
        'Assay'] + '/' + beluga_features_df['Cell type']

    # account for ablations
    keep_mask = get_keep_mask(args, beluga_features_df)

    input_features_df = beluga_features_df[keep_mask]
    input_features_df['cluster'] = labels
    input_features_df.to_csv(f'{args.out_dir}/all_feature_clusters.tsv', sep='\t')

    cluster_dir = f'{args.out_dir}/clusters'
    os.makedirs(cluster_dir, exist_ok=True)
    sizes_df = pd.DataFrame(columns=['size'])

    for i in np.unique(labels):
        cluster_df = input_features_df[input_features_df['cluster'] == i]
        cluster_df.to_csv(f'{cluster_dir}/cluster_{i}.tsv', sep='\t')
        sizes_df.loc[f'cluster_{i}'] = cluster_df.shape[0]

    sizes_df = sizes_df.sort_values(by='size', ascending=False)
    sizes_df.to_csv(f'{args.out_dir}/cluster_sizes.tsv', sep='\t')


if __name__ == '__main__':
    main()
