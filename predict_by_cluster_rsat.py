# -*- coding: utf-8 -*-
import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.cm import get_cmap
from six.moves import reduce

from cluster_utils import get_keep_mask

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--coorFile_chromatin', action="store", dest="coorFile_chromatin")
parser.add_argument('--geneFile', action="store",
                    dest="geneFile")
parser.add_argument('--snpEffectFilePattern', action="store", dest="snpEffectFilePattern",
                    help="SNP effect hdf5 filename pattern. Use SHIFT as placeholder for shifts.")
parser.add_argument('--model_save_file', action="store", dest="model_save_file",
                    help="Save file containing model to use for predictions")
parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                    help="tsv file denoting Beluga features")
parser.add_argument('--rsat_clusters_tab', action="store", dest="rsat_clusters_tab",
                    help="clusters_motif_names.tab")
parser.add_argument('--nfeatures', action="store",
                    dest="nfeatures", type=int, default=2002)
parser.add_argument('--fixeddist', action="store",
                    dest="fixeddist", default=0, type=int)
parser.add_argument('--maxshift', action="store",
                    dest="maxshift", type=int, default=800)
parser.add_argument('--batchSize', action="store",
                    dest="batchSize", type=int, default=500)
parser.add_argument('--splitIndex', action="store",
                    dest="splitIndex", type=int, default=0)
parser.add_argument('--splitFold', action="store",
                    dest="splitFold", type=int, default=10)
parser.add_argument('--threads', action="store", dest="threads",
                    type=int, default=16, help="Number of threads.")
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

parser.add_argument('-o', action="store", dest="out_dir")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

MOTIF_NOT_FOUND_CLUSTER_ID = -1

# Interpret SED scores
beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df['Assay'] + '/' + beluga_features_df['Cell type']

keep_mask, hgnc_df = get_keep_mask(beluga_features_df, args.no_tf_features, args.no_dnase_features,
                  args.no_histone_features, args.intersect_with_lambert, args.no_pol2, return_hgnc_df=True)

hgnc_df = hgnc_df[keep_mask]


# Read tab file from RSAT matrix-clustering
rsat_clusters_df = pd.read_csv(args.rsat_clusters_tab, sep='\t', header=None)

# Get cluster IDs for each assay
def preprocess_hocomoco_motifs(motif_list):
    processed_motif_list = []
    for motif in motif_list:
        processed_motif_list.append(motif.split("_")[0])
    return processed_motif_list

rsat_cluster_sets = rsat_clusters_df[1].str.upper().str.split(',').apply(preprocess_hocomoco_motifs).apply(set)

assay_clusters = []
motifs_not_found = set()
for assay in hgnc_df["Assay"]:
    assay_i_clusters = []
    for cluster_idx, motif_set in enumerate(rsat_cluster_sets.values):
        if assay in motif_set:
            assay_i_clusters.append(cluster_idx + 1)  # rsat cluster ids start from 1

    if len(assay_i_clusters) == 0:
        assay_clusters.append([MOTIF_NOT_FOUND_CLUSTER_ID])
        motifs_not_found.add(assay)
    else:
        assay_clusters.append(assay_i_clusters)

hgnc_df["cluster"] = assay_clusters
rsat_clusters_df = rsat_clusters_df.set_index(0)
rsat_clusters_df.loc["cluster_-1"] = ",".join(list(motifs_not_found))


def interpret_model(model, ref_features, alt_features):
    dump = model.get_dump()[0].strip('\n').split('\n')
    bias = float(dump[1])
    weights = np.array(list(map(float, dump[3:])))

    preds_per_feature = (weights * (alt_features - ref_features))  # omit bias term because of difference
    preds_per_feature = preds_per_feature.ravel()\
        .reshape(preds_per_feature.shape[0], 10, preds_per_feature.shape[1] // 10)\
        .transpose(0, 2, 1)  # (n_snps, n_chromatin_marks, n_features_per_mark)

    preds_per_feature = preds_per_feature.sum(axis=-1)  # sum over exponential basis function contributions
    preds_per_feature_proportion = preds_per_feature / preds_per_feature.sum(axis=-1, keepdims=True)

    # test = model.predict(xgb.DMatrix(alt_features)) - model.predict(xgb.DMatrix(ref_features))
    return preds_per_feature_proportion


def interpret_model_with_clusters_rsat(model, ref_features, alt_features, clusters):
    dump = model.get_dump()[0].strip('\n').split('\n')
    bias = float(dump[1])
    weights = np.array(list(map(float, dump[3:])))

    preds_per_feature = (weights * (alt_features - ref_features))  # omit bias term because of difference
    preds_per_feature = preds_per_feature.ravel()\
        .reshape(preds_per_feature.shape[0], 10, preds_per_feature.shape[1] // 10)\
        .transpose(0, 2, 1)  # (n_snps, n_chromatin_marks, n_features_per_mark)

    preds_per_feature_df = pd.DataFrame(preds_per_feature.reshape(preds_per_feature.shape[0], -1).T)  # (n_input_features, n_snps)

    # for each feature, add its contribution to all clusters that the feature belongs to
    cluster_contribs = {}
    for fi in range(preds_per_feature_df.shape[0]):
        feature_preds = preds_per_feature_df.iloc[fi]
        cluster_ids = clusters[fi // 10]  # 10 feature per mark
        for cluster_id in cluster_ids:
            cluster_contribs[cluster_id] = cluster_contribs.get(cluster_id, 0) + feature_preds

    cluster_contribs = pd.DataFrame(cluster_contribs)  # n_snps x n_clusters
    cluster_contribs_proportion = cluster_contribs.values / cluster_contribs.values.sum(axis=-1, keepdims=True)
    return cluster_contribs_proportion, cluster_contribs.columns


def compute_effects(snpeffects, ref_preds, alt_preds, snpdists, snpstrands, model, maxshift=800,
                    nfeatures=2002, batchSize=500, clusters=None):
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

    ref = np.zeros(n_snps)
    alt = np.zeros(n_snps)
    effect = np.zeros(n_snps)
    keep_mask = get_keep_mask(beluga_features_df, args.no_tf_features, args.no_dnase_features,
                              args.no_histone_features, args.intersect_with_lambert, args.no_pol2)
    keep_indices = np.nonzero(keep_mask)[0]
    n_marks = np.sum(keep_mask)
    preds_per_feature_proportion = np.zeros((n_snps, n_marks))

    num_clusters = len(set().union(*clusters))  # includes the "motif not found" cluster
    cluster_proportions = np.zeros((n_snps, num_clusters))

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

        # adjust for training on subset of tracks
        keep_mask = get_keep_mask(beluga_features_df, args.no_tf_features, args.no_dnase_features,
                  args.no_histone_features, args.intersect_with_lambert, args.no_pol2)
        keep_indices = np.nonzero(keep_mask)[0]
        n_marks = np.sum(keep_mask)
        alt_features = alt_features.reshape(alt_features.shape[0], 10, 2002)[:, :, keep_indices].reshape(
            alt_features.shape[0], -1)
        ref_features = ref_features.reshape(ref_features.shape[0], 10, 2002)[:, :, keep_indices].reshape(
            ref_features.shape[0], -1)
        diff = diff.reshape(diff.shape[0], 10, 2002)[:, :, keep_indices].reshape(
            diff.shape[0], -1)


        dtest_ref = xgb.DMatrix(diff * 0)
        dtest_alt = xgb.DMatrix(diff)

        dtest_ref_preds = xgb.DMatrix(ref_features)
        dtest_alt_preds = xgb.DMatrix(alt_features)

        effect[i * batchSize:(i + 1) * batchSize] = model.predict(dtest_ref) - \
                                                       model.predict(dtest_alt)

        ref[i * batchSize:(i + 1) * batchSize] = model.predict(dtest_ref_preds)
        alt[i * batchSize:(i + 1) * batchSize] = model.predict(dtest_alt_preds)

        preds_per_feature_proportion[i * batchSize:(i + 1) * batchSize, :] = \
            interpret_model(model, ref_features, alt_features)

        cluster_proportions[i * batchSize:(i + 1) * batchSize, :], cluster_proportions_columns = \
            interpret_model_with_clusters_rsat(model, ref_features, alt_features, clusters)

    return effect, ref, alt, preds_per_feature_proportion, cluster_proportions, cluster_proportions_columns

#load resources
model = xgb.Booster({'nthread': args.threads})
model.load_model(args.model_save_file.strip())

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


coor = pd.read_csv(args.coorFile_chromatin, sep='\t', header=None, comment='#')
# coor = coor.iloc[index_start:index_end, :]

#Fetch the distance to TSS information
def get_num_repeats(genes_df):
    repeats = [0]
    i = 0
    prev = None
    for _, row in genes_df.iterrows():
        curr = ':'.join(list(map(str, row.iloc[0:5])))  # '1:10690947:10690948:C:T'
        if prev is not None and curr != prev:
            repeats.append(0)
            i += 1
        repeats[i] += 1
        prev = curr
    return repeats


gene = pd.read_csv(args.geneFile,sep='\t',header=None,comment='#')

# The below assumes chromatin.py was used to generate the coor vcf, and make_closest_genes_file.py was used to generate the gene vcf

# deal with duplicates in original snps vcf
gene = gene.drop_duplicates(keep="first")
coor_mask = ~coor.duplicated(keep="first")
coor = coor[coor_mask]
ref_preds = np.array(ref_preds)[:, coor_mask, :]
alt_preds = np.array(alt_preds)[:, coor_mask, :]
snpEffects = np.array(snpEffects)[:, coor_mask, :]

# Get num repeats to match gene with chromatin vcf, then repeat the vcf entries and chromatin effects
repeats = get_num_repeats(gene)
coor_new = pd.DataFrame(np.repeat(coor.values, repeats, axis=0))
coor_new.columns = coor.columns
coor = coor_new

ref_preds = np.repeat(ref_preds, repeats=repeats, axis=1)
alt_preds = np.repeat(alt_preds, repeats=repeats, axis=1)
snpEffects = np.repeat(snpEffects, repeats=repeats, axis=1)

geneinds = np.arange(coor.shape[0])

if np.any(geneinds==-1):
    raise ValueError("Gene association file does not match the vcf file.")
if args.fixeddist == 0:
    dist = - np.asarray(gene.iloc[geneinds,-1])
else:
    dist = args.fixeddist
genename = np.asarray(gene.iloc[geneinds,-2])
strand = np.asarray(gene.iloc[geneinds,-3])

# compute expression effects
compute_effects_results = compute_effects(snpEffects, ref_preds, alt_preds,
                                          dist, strand,
                                          model, maxshift=maxshift,
                                          nfeatures=args.nfeatures,
                                          batchSize=args.batchSize,
                                          clusters=assay_clusters)

snpExpEffects, ref, alt, preds_per_feature_proportion, cluster_proportions, cluster_proportions_columns = compute_effects_results

#write output
snpExpEffects_df = coor
snpExpEffects_df['dist'] = dist
snpExpEffects_df['gene'] = genename
snpExpEffects_df['strand'] = strand

snpExpEffects_df = pd.concat([snpExpEffects_df.reset_index(),
                              pd.DataFrame(ref, columns=['REF']),
                              pd.DataFrame(alt, columns=['ALT']),
                              # pd.DataFrame(snpExpEffects, columns=['SED'])
                              pd.DataFrame(alt - ref, columns=['SED'])
                              ],
                             axis=1,
                             ignore_index=False)
snpExpEffects_df.to_csv(f'{args.out_dir}/sed.csv', header=True, sep='\t', index=False)
feature_contributions_df = pd.DataFrame(preds_per_feature_proportion.squeeze(), columns=hgnc_df['Assay'] + '/' + hgnc_df['Cell type'])

# Sort by magnitude of SNP effects
snpExpEffects_df_sorted = snpExpEffects_df.copy()
snpExpEffects_df_sorted['SED_MAGNITUDES'] = np.abs(snpExpEffects_df_sorted['SED'])
snpExpEffects_df_sorted = snpExpEffects_df_sorted.sort_values(by='SED_MAGNITUDES', axis=0, ascending=False)
snpExpEffects_df_sorted.to_csv(f'{args.out_dir}/sed_sorted_by_magnitude.tsv', header=True, sep='\t', index=False)

# Sort by SAD magnitude proportion
snpExpEffects_df_sorted = snpExpEffects_df.copy()
snpExpEffects_df_sorted['SED_PROPORTION'] = np.abs(snpExpEffects_df_sorted['SED'] / ((snpExpEffects_df_sorted['REF'] + snpExpEffects_df_sorted['ALT']) / 2))
snpExpEffects_df_sorted = snpExpEffects_df_sorted.sort_values(by='SED_PROPORTION', axis=0, ascending=False)
snpExpEffects_df_sorted.to_csv(f'{args.out_dir}/sed_sorted_by_proportion.tsv', header=True, sep='\t', index=False)

sed_feature_contributions_df = snpExpEffects_df.copy()
sed_feature_contributions_df['SED_PROPORTION'] = np.abs(sed_feature_contributions_df['SED'] / ((sed_feature_contributions_df['REF'] + sed_feature_contributions_df['ALT']) / 2))
sed_feature_contributions_df = pd.concat([sed_feature_contributions_df, feature_contributions_df], axis=1)
sed_feature_contributions_df = sed_feature_contributions_df.sort_values(by='SED_PROPORTION', axis=0, ascending=False).reset_index(drop=True)
sed_feature_contributions_df.to_csv(f'{args.out_dir}/sed_sorted_by_proportion_with_contribs.tsv', header=True, sep='\t', index=False)

# Plotting
# TODO: Plot top k genes
# TODO: Plot top m features by absolute value
cluster_proportions = cluster_proportions.squeeze()
cluster_proportions_df = pd.DataFrame(cluster_proportions,
                                      columns=[f'cluster_{idx}' for idx in cluster_proportions_columns])
sed_cluster_proportions_df = snpExpEffects_df.copy()
sed_cluster_proportions_df['SED_PROPORTION'] = np.abs(sed_cluster_proportions_df['SED'] / ((sed_cluster_proportions_df['REF'] + sed_cluster_proportions_df['ALT']) / 2))
sed_cluster_proportions_df = pd.concat([sed_cluster_proportions_df, cluster_proportions_df], axis=1)
sed_cluster_proportions_df = sed_cluster_proportions_df.sort_values(by='SED_PROPORTION', axis=0, ascending=False).reset_index(drop=True)
sed_cluster_proportions_df.to_csv(f'{args.out_dir}/cluster_contribs.tsv', header=True, sep='\t', index=False)

cluster_proportions = cluster_proportions.squeeze()
cluster_figures_dir = f'{args.out_dir}/cluster_figures'
os.makedirs(cluster_figures_dir, exist_ok=True)

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


# Save cluster tab df
rsat_clusters_df.to_csv(f"{args.out_dir}/rsat_clusters.tsv", sep="\t", header=None)
