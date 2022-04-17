# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Cluster and visualize SVD-transformed data')
    parser.add_argument('svd_transform_dir')
    parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                        help="tsv file denoting Beluga features")
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

    # k-means elbow plot
    # objectives = []
    # for k in tqdm(range(2, 200, 5)):
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans.fit(X)
    #     objectives.append(kmeans.inertia_)
    #
    # plt.figure()
    # plt.plot(objectives)
    # plt.show()

    # Clustering with chosen k
    print("clustering...")
    k = 100  # choose k to use
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    labels = kmeans.labels_

    # t-SNE
    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(X)

    plt.figure()
    for i in np.unique(labels):
        X_plot = X_embedded[labels == i]
        plt.scatter(X_plot[:, 0], X_plot[:, 1], label=f'cluster {i}')
    plt.show()
    print('hi')


    # Save clustering
    beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
    beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df[
        'Assay'] + '/' + beluga_features_df['Cell type']

    input_features_df = beluga_features_df
    input_features_df['cluster'] = labels
    input_features_df.to_csv(f'{args.out_dir}/all_feature_clusters.tsv', sep='\t')

    cluster_dir = f'{args.out_dir}/clusters'
    os.makedirs(cluster_dir, exist_ok=True)
    sizes_df = pd.DataFrame(columns=['size'])
    for i in range(k):
        cluster_df = input_features_df[input_features_df['cluster'] == i]
        cluster_df.to_csv(f'{cluster_dir}/cluster_{i}.tsv', sep='\t')
        sizes_df.loc[f'cluster_{i}'] = cluster_df.shape[0]

    sizes_df = sizes_df.sort_values(by='size', ascending=False)
    sizes_df.to_csv(f'{args.out_dir}/cluster_sizes.tsv', sep='\t')

    # # print("Plotting dendrogram...")

if __name__ == '__main__':
    main()
