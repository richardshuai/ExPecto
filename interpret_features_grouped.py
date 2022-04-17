import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from joblib import dump, load
import itertools
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--clustering_joblib', action='store', dest='clustering_joblib',
                    type=str, default=None, help="joblib dump for clustering model. If None, recomputes the "
                                                 "hierarchical clustering given the inputFile training data.")
parser.add_argument('--clustering_with_distances', action="store_true", default=False)
parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                    help="tsv file denoting Beluga features")
parser.add_argument('--targetIndex', action="store",
                    dest="targetIndex", type=int)
parser.add_argument('--expFile', action="store", dest="expFile")
parser.add_argument('--inputFile', action="store",
                    dest="inputFile", default='./resources/Xreducedall.2002.npy')
parser.add_argument('--annoFile', action="store",
                    dest="annoFile", default='./resources/geneanno.csv')
parser.add_argument('--filterStr', action="store",
                    dest="filterStr", type=str, default="all")
parser.add_argument('--pseudocount', action="store",
                    dest="pseudocount", type=float, default=0.0001)
parser.add_argument('--out_dir', type=str, default='interpret_features')
args = parser.parse_args()

# Reproducibility
np.random.seed(0)

# Make model output dir
os.makedirs(args.out_dir, exist_ok=True)

# read resources
Xreducedall = np.load(args.inputFile)
geneanno = pd.read_csv(args.annoFile)

if args.filterStr == 'pc':
    filt = np.asarray(geneanno.iloc[:, -1] == 'protein_coding')
elif args.filterStr == 'lincRNA':
    filt = np.asarray(geneanno.iloc[:, -1] == 'lincRNA')
elif args.filterStr == 'all':
    filt = np.asarray(geneanno.iloc[:, -1] != 'rRNA')
else:
    raise ValueError('filterStr has to be one of all, pc, and lincRNA')

geneexp = pd.read_csv(args.expFile)
print(f"Cell type: {geneexp.columns[args.targetIndex]}")

filt = filt * \
    np.isfinite(np.asarray(
        np.log(geneexp.iloc[:, args.targetIndex] + args.pseudocount)))

# training
trainind = np.asarray(geneanno['seqnames'] != 'chrX') * np.asarray(geneanno['seqnames'] != 'chrY') \
           * np.asarray(geneanno['seqnames'] != 'chr8')

testind = np.asarray(geneanno['seqnames'] == 'chr8')

X_train = Xreducedall[trainind * filt, :]

X_train_grouped = X_train.T.reshape(10, 2002, -1).transpose(1, 2, 0).reshape(2002, -1)  # num_marks x num_genes * 10 features per track

X_test = Xreducedall[testind * filt, :]

y_train = np.asarray(np.log(geneexp.iloc[trainind * filt, args.targetIndex] + args.pseudocount))
y_test = np.asarray(np.log(geneexp.iloc[testind * filt, args.targetIndex] + args.pseudocount))

# Hierarchical clustering of features based on training dataset
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

if args.clustering_joblib is None:
    if args.clustering_with_distances:
        print("Clustering and saving model to joblib...")
        clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X_train_grouped)
        dump(clustering, f'{args.out_dir}/clustering_with_distances.joblib')
    else:
        print("Clustering complete tree and caching...")
        clustering = AgglomerativeClustering(compute_full_tree=True, memory=f'{args.out_dir}/cache').fit(X_train_grouped)
        dump(clustering, f'{args.out_dir}/clustering_cached.joblib')
else:
    print(f"Loading clustering model from {args.clustering_joblib}...")
    clustering = load(args.clustering_joblib)

beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df[
    'Assay'] + '/' + beluga_features_df['Cell type']

# Determine number of clusters
# max_n_clusters = 500
# silhouette_scores = []
# for k in tqdm(range(2, max_n_clusters)):
#     clustering.set_params(n_clusters=k).fit(X_train_grouped)
#     clusters = clustering.labels_
#     silhouette_scores.append(silhouette_score(X_train_grouped, clusters))
#
# plt.figure()
# plt.plot(np.arange(2, len(silhouette_scores) + 2), silhouette_scores)
# plt.title("Silhouette score vs. num_clusters")
# plt.xlabel('Number of clusters')
# plt.ylabel('Silhouette score')
# plt.savefig(f'{args.out_dir}/silhouette_scores.png', dpi=300)
# plt.show()

# silhouette_scores = []
# for k in tqdm(range(100, 115)):
#     clustering.set_params(n_clusters=k).fit(X_train_grouped)
#     clusters = clustering.labels_
#     silhouette_scores.append(silhouette_score(X_train_grouped, clusters))

# Get cluster for each feature
n_clusters = 108 + 2  # TODO: Set this manually
# n_clusters = np.argmax(silhouette_scores) + 2  # max silhouette score
clustering.set_params(n_clusters=n_clusters).fit(X_train_grouped)
clusters = clustering.labels_

# Save clusters to file
input_features_df = beluga_features_df
input_features_df['cluster'] = clusters.ravel()
input_features_df.to_csv(f'{args.out_dir}/all_feature_clusters.tsv', sep='\t')

cluster_dir = f'{args.out_dir}/clusters'
os.makedirs(cluster_dir, exist_ok=True)
sizes_df = pd.DataFrame(columns=['size'])
for i in range(n_clusters):
    cluster_df = input_features_df[input_features_df['cluster'] == i]
    cluster_df.to_csv(f'{cluster_dir}/cluster_{i}.tsv', sep='\t')
    sizes_df.loc[f'cluster_{i}'] = cluster_df.shape[0]

sizes_df = sizes_df.sort_values(by='size', ascending=False)
sizes_df.to_csv(f'{args.out_dir}/cluster_sizes.tsv', sep='\t')

# # print("Plotting dendrogram...")
# # plt.title("Hierarchical Clustering Dendrogram")
# # # plot the top three levels of the dendrogram
# # plot_dendrogram(clustering, truncate_mode="level", p=3)
# # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# # plt.show()
#
