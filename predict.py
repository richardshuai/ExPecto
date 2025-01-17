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
parser.add_argument('--model_save_file', action="store", dest="model_save_file",
                    help="Save file containing model to use for predictions")
parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                    help="tsv file denoting Beluga features")
parser.add_argument('--coorFile_chromatin', action="store", dest="coorFile_chromatin")
parser.add_argument('--geneFile', action="store",
                    dest="geneFile")
parser.add_argument('--snpEffectFilePattern', action="store", dest="snpEffectFilePattern",
                    help="SNP effect hdf5 filename pattern. Use SHIFT as placeholder for shifts.")
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

# Interpret SED scores
beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df['Assay'] + '/' + beluga_features_df['Cell type']

keep_mask = get_keep_mask(beluga_features_df, args.no_tf_features, args.no_dnase_features,
                          args.no_histone_features, args.intersect_with_lambert, args.no_pol2)


def compute_effects(snpeffects, ref_preds, alt_preds, snpdists, snpstrands, model, maxshift=800, nfeatures=2002, batchSize=500):
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

    return effect, ref, alt

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
# coor = coor.iloc[index_start:index_end,:]

#Fetch the distance to TSS information
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
strand= np.asarray(gene.iloc[geneinds,-3])

# compute expression effects
snpExpEffects, ref, alt, = compute_effects(snpEffects, ref_preds, alt_preds,
                                                dist, strand,
                                                model, maxshift=maxshift,
                                                nfeatures=args.nfeatures,
                                                batchSize=args.batchSize)
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
snpExpEffects_df.to_csv(f'{args.out_dir}/sed.tsv', header=True, sep='\t', index=False)

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
