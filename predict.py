# -*- coding: utf-8 -*-
"""Predict variant expression effects

This script takes the predicted chromatin effects computed by chromatin.py and
expression model file list, and predicts expression effects in all models provided
in the model list.

Example:
        $ python predict.py --coorFile ./example/example.vcf --geneFile ./example/example.vcf.bed.sorted.bed.closestgene --snpEffectFilePattern ./example/example.vcf.shift_SHIFT.diff.h5 --modelList ./resources/modellist --output output.csv
For very large input files use the split functionality to distribute the
prediction into multiple runs. For example, `--splitFlag --splitIndex 0 --splitFold 10`
will divide the input into 10 chunks and process only the first chunk.

"""
import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
import h5py
from six.moves import reduce
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--coorFile', action="store", dest="coorFile")
parser.add_argument('--geneFile', action="store",
                    dest="geneFile")
parser.add_argument('--snpEffectFilePattern', action="store", dest="snpEffectFilePattern",
                    help="SNP effect hdf5 filename pattern. Use SHIFT as placeholder for shifts.")
parser.add_argument('--modelList', action="store", dest="modelList",
                    help="A list of paths of binary xgboost model files (if end with .list) or a combined model file (if ends with .csv).")
parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                    help="tsv file denoting Beluga features")
parser.add_argument('--nfeatures', action="store",
                    dest="nfeatures", type=int, default=2002)
parser.add_argument('-o', action="store", dest="out_dir")
parser.add_argument('--fixeddist', action="store",
                    dest="fixeddist", default=0, type=int)
parser.add_argument('--maxshift', action="store",
                    dest="maxshift", type=int, default=800)
parser.add_argument('--batchSize', action="store",
                    dest="batchSize", type=int, default=500)
parser.add_argument('--splitFlag', action="store_true", default=False)
parser.add_argument('--splitIndex', action="store",
                    dest="splitIndex", type=int, default=0)
parser.add_argument('--splitFold', action="store",
                    dest="splitFold", type=int, default=10)
parser.add_argument('--threads', action="store", dest="threads",
                    type=int, default=16, help="Number of threads.")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

input_features_df = pd.read_csv('output_dir/interpret_features_grouped/all_feature_clusters.tsv', sep='\t', index_col=0)
clusters = input_features_df['cluster']


def interpret_model(model, ref_features, alt_features):
    dump = model.get_dump()[0].strip('\n').split('\n')
    bias = float(dump[1])
    weights = np.array(list(map(float, dump[3:])))
    preds_per_feature = (weights * (alt_features - ref_features))  # omit bias term because of difference
    preds_per_feature = preds_per_feature.ravel()\
        .reshape(preds_per_feature.shape[0], 10, 2002)\
        .transpose(0, 2, 1)  # (n_snps, n_chromatin_marks, n_features_per_mark)

    preds_per_feature = preds_per_feature.sum(axis=-1)  # sum over exponential basis function contributions
    preds_per_feature_proportion = preds_per_feature / preds_per_feature.sum(axis=-1, keepdims=True)

    # test = model.predict(xgb.DMatrix(alt_features)) - model.predict(xgb.DMatrix(ref_features))
    return preds_per_feature_proportion


def interpret_model_with_clusters(model, ref_features, alt_features, clusters):
    # TODO: Normalize cluster contribs by size of cluster? to get average contribution of feature in cluster.
    # TODO: Just change it to mean() to do this, but I don't think it makes sense to take the avg since it penalizes redundant features.
    dump = model.get_dump()[0].strip('\n').split('\n')
    bias = float(dump[1])
    weights = np.array(list(map(float, dump[3:])))
    preds_per_feature = (weights * (alt_features - ref_features))  # omit bias term because of difference
    preds_per_feature = preds_per_feature.ravel()\
        .reshape(preds_per_feature.shape[0], 10, 2002)\
        .transpose(0, 2, 1)  # (n_snps, n_chromatin_marks, n_features_per_mark)

    preds_per_feature_df = pd.DataFrame(preds_per_feature.reshape(preds_per_feature.shape[0], -1).T)  # (n_input_features, n_snps)
    preds_per_feature_df['cluster'] = clusters
    cluster_contribs = preds_per_feature_df.groupby('cluster').sum().values.T

    cluster_contribs_proportion = cluster_contribs / cluster_contribs.sum(axis=-1, keepdims=True)

    return cluster_contribs_proportion


def compute_effects(snpeffects, ref_preds, alt_preds, snpdists, snpstrands, all_models, maxshift=800, nfeatures=2002, batchSize=500,old_format=False):
    """Compute expression effects (log fold-change).

    Args:
        snpeffects: list of chromatin effect numpy arrays
        snpdists:  integer array or pandas Series representing distances to TSS
        snpstrands: string array or pandas Series containing only '+' and '-'s
                     representing the strand of the TSS for each variant
        all_models: list of ExPecto model files.
        maxshift:  maximum shift distance for chromatin effects.
        nfeatures: number of chromatin/epigenomic features.
        batchSize: batch size when computing ExPecto predictions.

    Returns:
        numpy array of size num_variants x num_models. Each value represents
        predicted log fold-change
    """
    snpdists = snpdists * ((snpstrands == '+') * 2 - 1)
    Xreducedall_diffs = [np.vstack([
    np.exp(-0.01 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
    np.exp(-0.02 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
    np.exp(-0.05 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
    np.exp(-0.1 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
    np.exp(-0.2 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) <= 0),
    np.exp(-0.01 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
    np.exp(-0.02 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
    np.exp(-0.05 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
    np.exp(-0.1 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0),
    np.exp(-0.2 * np.floor(np.abs((snpdists + dist * ((snpstrands == '+') * 2 - 1)
           ) / 200.0))) * ((snpdists + dist * ((snpstrands == '+') * 2 - 1)) >= 0)
     ]).T for dist in [0, ] + list(range(-200, -maxshift - 1, -200)) + list(range(200, maxshift + 1, 200))]
    n_snps = len(snpdists)

    ref = np.zeros((n_snps, len(all_models)))
    alt = np.zeros((n_snps, len(all_models)))
    effect = np.zeros((n_snps, len(all_models)))
    preds_per_feature_proportion = np.zeros((n_snps, nfeatures, len(all_models)))
    cluster_proportions = np.zeros((n_snps, len(np.unique(clusters)), len(all_models)))

    for i in range(int( (n_snps - 1) / batchSize) + 1):
        print("Processing " + str(i) + "th batch of "+str(batchSize))
        # compute gene expression change with models
        diff = reduce(lambda x, y: x + y, [np.tile(np.asarray(snpeffects[j][i * batchSize:(i + 1) * batchSize, :]), 10)
                                 * np.repeat(Xreducedall_diffs[j][i * batchSize:(i + 1) * batchSize, :], nfeatures, axis=1) for j in range(len(Xreducedall_diffs))])
        # x = np.array(snpeffects)
        # y = np.array(Xreducedall_diffs)
        # y.shape
        # (9, 423, 10)
        # x.shape
        # (9, 423, 2002)
        # np.sum(x[:, :, None, :] * y[:, :, :, None], axis=0).reshape(x.shape[1], -1)
        ref_features = reduce(lambda x, y: x + y, [np.tile(np.asarray(ref_preds[j][i * batchSize:(i + 1) * batchSize, :]), 10)
                                 * np.repeat(Xreducedall_diffs[j][i * batchSize:(i + 1) * batchSize, :], nfeatures, axis=1) for j in range(len(Xreducedall_diffs))])

        alt_features = reduce(lambda x, y: x + y, [np.tile(np.asarray(alt_preds[j][i * batchSize:(i + 1) * batchSize, :]), 10)
                                 * np.repeat(Xreducedall_diffs[j][i * batchSize:(i + 1) * batchSize, :], nfeatures, axis=1) for j in range(len(Xreducedall_diffs))])

        if old_format:
            # backward compatibility
            diff = np.concatenate([np.zeros((diff.shape[0], 10, 1)), diff.reshape(
                (-1, 10, 2002))], axis=2).reshape((-1, 20030))
        dtest_ref = xgb.DMatrix(diff * 0)
        dtest_alt = xgb.DMatrix(diff)

        dtest_ref_preds = xgb.DMatrix(ref_features)
        dtest_alt_preds = xgb.DMatrix(alt_features)

        for j in range(len(all_models)):
            model = all_models[j]
            effect[i * batchSize:(i + 1) * batchSize, j] = model.predict(dtest_ref) - \
                                                           model.predict(dtest_alt)

            ref[i * batchSize:(i + 1) * batchSize, j] = model.predict(dtest_ref_preds)
            alt[i * batchSize:(i + 1) * batchSize, j] = model.predict(dtest_alt_preds)

            preds_per_feature_proportion[i * batchSize:(i + 1) * batchSize, :, j] = \
                interpret_model(model, ref_features, alt_features)

            cluster_proportions[i * batchSize:(i + 1) * batchSize, :, j] = \
                interpret_model_with_clusters(model, ref_features, alt_features, clusters)


    return effect, ref, alt, preds_per_feature_proportion, cluster_proportions

#load resources
modelList = pd.read_csv(args.modelList,sep='\t',header=0)
models = []
for file in modelList['ModelName']:
        bst = xgb.Booster({'nthread': args.threads})
        bst.load_model(file.strip())
        models.append(bst)

# backward compatibility with earlier model format
if len(bst.get_dump()[0].split('\n')) == 20034:
    old_format = True
else:
    old_format = False


#load input data
maxshift = int(args.maxshift)
snpEffects = []
ref_preds = []
alt_preds = []
for shift in [str(n) for n in [0, ] + list(range(-200, -maxshift - 1, -200)) + list(range(200, maxshift + 1, 200))]:
    h5f_diff = h5py.File(args.snpEffectFilePattern.replace(
        'SHIFT', shift), 'r')['diff']

    h5f_ref = h5py.File(args.snpEffectFilePattern.replace(
        'SHIFT', shift), 'r')['ref']

    h5f_alt = h5py.File(args.snpEffectFilePattern.replace(
        'SHIFT', shift), 'r')['alt']

    if args.splitFlag:
        index_start = int((args.splitIndex - 1) *
                          np.ceil(float(h5f_diff.shape[0] / 2) / args.splitFold))
        index_end = int(np.minimum(
            (args.splitIndex) * np.ceil(float(h5f_diff.shape[0] / 2) / args.splitFold), (h5f_diff.shape[0] / 2)))
    else:
        index_start = 0
        index_end = int(h5f_diff.shape[0] / 2)

    snp_temp = (np.asarray(h5f_diff[index_start:index_end, :]) + np.asarray(h5f_diff[index_start + int(h5f_diff.shape[0] / 2):index_end + int(h5f_diff.shape[0] / 2), :])) / 2.0
    snp_temp_ref = (np.asarray(h5f_ref[index_start:index_end, :]) + np.asarray(
        h5f_ref[index_start + int(h5f_ref.shape[0] / 2):index_end + int(h5f_ref.shape[0] / 2), :])) / 2.0
    snp_temp_alt = (np.asarray(h5f_alt[index_start:index_end, :]) + np.asarray(
        h5f_alt[index_start + int(h5f_alt.shape[0] / 2):index_end + int(h5f_alt.shape[0] / 2), :])) / 2.0

    snpEffects.append(snp_temp)
    ref_preds.append(snp_temp_ref)
    alt_preds.append(snp_temp_alt)


coor = pd.read_csv(args.coorFile,sep='\t',header=None,comment='#')
coor = coor.iloc[index_start:index_end,:]

#Fetch the distance to TSS information

def add_multiplicity_suffixes(arr):
    """
    Takes in array and appends suffixes _i where i is the number of times the value has shown up previously
    in the array. Used for matching below.
    """
    suffixed_array = []
    counts = {}
    for x in arr:
        suffixed_x = x + f'_{counts.get(x, 0)}'
        counts[x] = counts.get(x, 0) + 1
        suffixed_array.append(suffixed_x)

    return np.array(suffixed_array)

gene = pd.read_csv(args.geneFile,sep='\t',header=None,comment='#')
geneinds = pd.match(add_multiplicity_suffixes(coor.iloc[:,0].map(str).str.replace('chr','')+' '+coor.iloc[:,1].map(str)),
            add_multiplicity_suffixes(gene.iloc[:,0].map(str).str.replace('chr','')+' '+gene.iloc[:,2].map(str)))

if np.any(geneinds==-1):
    raise ValueError("Gene association file does not match the vcf file.")
if args.fixeddist == 0:
    dist = - np.asarray(gene.iloc[geneinds,-1])
else:
    dist = args.fixeddist
genename = np.asarray(gene.iloc[geneinds,-2])
strand= np.asarray(gene.iloc[geneinds,-3])

# compute expression effects
snpExpEffects, ref, alt, preds_per_feature_proportion, cluster_proportions = compute_effects(snpEffects, ref_preds, alt_preds,
                                                                        dist, strand,
                                                                        models, maxshift=maxshift,
                                                                        nfeatures=args.nfeatures,
                                                                        batchSize=args.batchSize, old_format=old_format)
#write output
snpExpEffects_df = coor
snpExpEffects_df['dist'] = dist
snpExpEffects_df['gene'] = genename
snpExpEffects_df['strand'] = strand
# snpExpEffects_df = pd.concat([snpExpEffects_df.reset_index(),
#                               pd.DataFrame(ref, columns=list(map(lambda x: x + '_REF', modelList.iloc[:, 1]))),
#                               pd.DataFrame(alt, columns=list(map(lambda x: x + '_ALT', modelList.iloc[:, 1]))),
#                               pd.DataFrame(snpExpEffects, columns=list(map(lambda x: x + '_SED', modelList.iloc[:, 1])))
#                               ],
#                              axis=1,
#                              ignore_index=False)

snpExpEffects_df = pd.concat([snpExpEffects_df.reset_index(),
                              pd.DataFrame(ref, columns=['REF']),
                              pd.DataFrame(alt, columns=['ALT']),
                              # pd.DataFrame(snpExpEffects, columns=['SED'])
                              pd.DataFrame(alt - ref, columns=['SED'])
                              ],
                             axis=1,
                             ignore_index=False)
snpExpEffects_df.to_csv(f'{args.out_dir}/sed.csv', header=True, sep='\t', index=False)

# Sort by magnitude of SNP effects
snpExpEffects_df_sorted = snpExpEffects_df.copy()
snpExpEffects_df_sorted['SED_MAGNITUDES'] = np.abs(snpExpEffects_df_sorted['SED'])
snpExpEffects_df_sorted = snpExpEffects_df_sorted.sort_values(by='SED_MAGNITUDES', axis=0, ascending=False)
snpExpEffects_df_sorted.to_csv(f'{args.out_dir}/sed_sorted_by_magnitude.csv', header=True, sep='\t', index=False)

# Sort by SAD magnitude proportion
snpExpEffects_df_sorted = snpExpEffects_df.copy()
snpExpEffects_df_sorted['SED_PROPORTION'] = np.abs(snpExpEffects_df_sorted['SED'] / ((snpExpEffects_df_sorted['REF'] + snpExpEffects_df_sorted['ALT']) / 2))
snpExpEffects_df_sorted = snpExpEffects_df_sorted.sort_values(by='SED_PROPORTION', axis=0, ascending=False)
snpExpEffects_df_sorted.to_csv(f'{args.out_dir}/sed_sorted_by_proportion.csv', header=True, sep='\t', index=False)

# Interpret SED scores
beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df['Assay'] + '/' + beluga_features_df['Cell type']
feature_contributions_df = pd.DataFrame(preds_per_feature_proportion.squeeze(), columns=beluga_features_df['Assay type + assay + cell type'])

sed_feature_contributions_df = snpExpEffects_df.copy()
sed_feature_contributions_df['SED_PROPORTION'] = np.abs(sed_feature_contributions_df['SED'] / ((sed_feature_contributions_df['REF'] + sed_feature_contributions_df['ALT']) / 2))
sed_feature_contributions_df = pd.concat([sed_feature_contributions_df, feature_contributions_df], axis=1)
sed_feature_contributions_df = sed_feature_contributions_df.sort_values(by='SED_PROPORTION', axis=0, ascending=False).reset_index(drop=True)
sed_feature_contributions_df.to_csv(f'{args.out_dir}/sed_sorted_by_proportion_with_contribs.csv', header=True, sep='\t', index=False)

# Plotting
# TODO: Plot top k genes
# # TODO: Plot top m features by absolute value
# k = 10
# m = 10
#
# figures_dir = f'{args.out_dir}/figures'
# os.makedirs(figures_dir, exist_ok=True)
#
# for i, row in sed_feature_contributions_df.iterrows():
#     if i == k:
#         break
#     feature_contribs = row.iloc[16:]
#     top_feature_contribs = feature_contribs.iloc[np.argsort(feature_contribs.apply(abs))[::-1][:m]]
#
#     plt.figure(figsize=(6.4, 8))
#
#     cmap = get_cmap("Set3")
#     colors = (cmap.colors * int(np.ceil(m / len(cmap.colors))))[:m]
#     plt.bar(range(m), top_feature_contribs, edgecolor='black', color=colors)
#     rsid, gene = row.iloc[3], row.iloc[10]
#     plt.title(f"{rsid} effect on {gene} by epigenomic feature contributions")
#
#     handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in range(m)]
#     labels = top_feature_contribs.index
#     plt.legend(handles, labels, bbox_to_anchor=(-0.1, -0.15), loc='upper left', ncol=1, fontsize=10)
#     plt.tight_layout()
#     plt.savefig(f"{figures_dir}/{rsid}_{gene}.png", bbox_inches="tight", dpi=300)
#     plt.show()


# Plotting
# TODO: Plot top k genes
# TODO: Plot top m features by absolute value
cluster_proportions = cluster_proportions.squeeze()
cluster_proportions_df = pd.DataFrame(cluster_proportions,
                                      columns=[f'cluster_{idx}' for idx in range(cluster_proportions.shape[1])])
sed_cluster_proportions_df = snpExpEffects_df.copy()
sed_cluster_proportions_df['SED_PROPORTION'] = np.abs(sed_cluster_proportions_df['SED'] / ((sed_cluster_proportions_df['REF'] + sed_cluster_proportions_df['ALT']) / 2))
sed_cluster_proportions_df = pd.concat([sed_cluster_proportions_df, cluster_proportions_df], axis=1)
sed_cluster_proportions_df = sed_cluster_proportions_df.sort_values(by='SED_PROPORTION', axis=0, ascending=False).reset_index(drop=True)
sed_cluster_proportions_df.to_csv(f'{args.out_dir}/cluster_contribs.csv', header=True, sep='\t', index=False)

k = 10

cluster_proportions = cluster_proportions.squeeze()
cluster_figures_dir = f'{args.out_dir}/cluster_figures'
os.makedirs(cluster_figures_dir, exist_ok=True)


# for i, row in sed_feature_contributions_df.iterrows():
#     if i == k:
#         break
#     feature_contribs = row.iloc[16:]
#     top_feature_contribs = feature_contribs.iloc[np.argsort(feature_contribs.apply(abs))[::-1][:m]]
#
#     plt.figure(figsize=(6.4, 8))

#
#     plt.figure(figsize=(6.4, 8))
#
#     cmap = get_cmap("Set3")
#     colors = (cmap.colors * int(np.ceil(m / len(cmap.colors))))[:m]
#     plt.bar(range(m), top_feature_contribs, edgecolor='black', color=colors)
#     rsid, gene = row.iloc[3], row.iloc[10]
#     plt.title(f"{rsid} effect on {gene} by epigenomic feature contributions")
#
#     handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in range(m)]
#     labels = top_feature_contribs.index
#     plt.legend(handles, labels, bbox_to_anchor=(-0.1, -0.15), loc='upper left', ncol=1, fontsize=10)
#     plt.tight_layout()
#     plt.savefig(f"{figures_dir}/{rsid}_{gene}.png", bbox_inches="tight", dpi=300)
#     plt.show()


k = 10  # num_snps to plot
m = 10  # num_clusters to plot
for i, row in sed_cluster_proportions_df.iterrows():
    if i == k:
        break
    cluster_contribs = row.iloc[16:]
    top_cluster_contribs = cluster_contribs.iloc[np.argsort(cluster_contribs.apply(abs))[::-1][:m]]

    plt.figure(figsize=(6.4, 8))

    cmap = get_cmap("Set3")
    colors = (cmap.colors * int(np.ceil(m / len(cmap.colors))))[:m]
    plt.bar(range(m), top_cluster_contribs, edgecolor='black', color=colors)
    rsid, gene = sed_feature_contributions_df.iloc[i, 3], sed_feature_contributions_df.iloc[i, 10]
    plt.title(f"{rsid} effect on {gene} by epigenomic feature contributions")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in range(10)]
    labels = top_cluster_contribs.index
    plt.legend(handles, labels, bbox_to_anchor=(-0.1, -0.15), loc='upper left', ncol=1, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{cluster_figures_dir}/{rsid}_{gene}.png", bbox_inches="tight", dpi=300)
    plt.show()

