# -*- coding: utf-8 -*-
import argparse
import xgboost as xgb
import pandas as pd
import numpy as np
import h5py
from six.moves import reduce
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import glob
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--coorFile', action="store", dest="coorFile")
    parser.add_argument('--geneFile', action="store", dest="geneFile")
    parser.add_argument('--snpEffectFilePattern', action="store", dest="snpEffectFilePattern",
                        help="SNP effect hdf5 filename pattern. Use SHIFT as placeholder for shifts.")
    parser.add_argument('--bootstrap_model_dir', action="store", dest="bootstrap_model_dir", type=str,
                        help='Directory containing bootstrapped models')
    parser.add_argument('--main_model', action="store", dest="main_model", type=str,
                        help='Main model fit on X_train without bootstrapping')
    parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                        help="tsv file denoting Beluga features")
    parser.add_argument('--nfeatures', action="store",
                        dest="nfeatures", type=int, default=2002)
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
    parser.add_argument('-o', action="store", dest="out_dir")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load resources
    model_files = natsorted(glob.glob(f'{args.bootstrap_model_dir}/*/*.save'))
    models = [load_model(model_file, args.threads) for model_file in model_files][:800]
    input_features_df = pd.read_csv('./output_dir/interpret_features/all_feature_clusters.tsv', sep='\t', index_col=0)

    # get model coefficient standard errors
    all_weights, all_biases = zip(*[get_model_coeffs(model) for model in models])
    all_weights, all_biases = np.vstack(all_weights), np.vstack(all_biases)

    se_weight = np.std(all_weights, axis=0, ddof=1)
    se_bias = np.std(all_biases, axis=0, ddof=1)

    # Most significant features (naive)
    # Get main model...?
    main_model = load_model(args.main_model, args.threads)
    weights, biases = get_model_coeffs(main_model)

    input_features_df['z_score'] = weights / se_weight
    input_features_df['temp'] = np.abs(input_features_df['z_score'])
    input_features_df = input_features_df.sort_values(by='temp', axis=0, ascending=False).reset_index(drop=True)
    input_features_df = input_features_df.drop('temp', axis=1)
    input_features_df.to_csv(f'{args.out_dir}/input_features_sorted_by_zscore.csv', sep='\t')

    # TODO: Bootstrapped confidence intervals? mean prediction?
    # TODO: Plot sampling distribution for highest coefficient of variation...
    # TODO: Table of most significant features ordered by p-values

    k = 10
    coeff_of_vars = se_weight / np.abs(np.mean(all_weights, axis=0))
    top_coeffs_i = np.argsort(coeff_of_vars)[-k:]

    for i in top_coeffs_i[::-1]:
        plt.figure()
        plt.hist(all_weights[:, i])
        plt.show()


def load_model(model_file, nthread):
    bst = xgb.Booster({'nthread': nthread})
    bst.load_model(model_file.strip())
    return bst


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


def get_model_coeffs(model):
    dump = model.get_dump()[0].strip('\n').split('\n')
    bias = float(dump[1])
    weights = np.array(list(map(float, dump[3:])))
    return weights, bias


def interpret_model(model, ref_features, alt_features):
    weights, bias = get_model_coeffs(model)
    preds_per_feature = (weights * (alt_features - ref_features))  # omit bias term because of difference
    preds_per_feature = preds_per_feature.ravel()\
        .reshape(preds_per_feature.shape[0], 10, 2002)\
        .transpose(0, 2, 1)  # (n_snps, n_chromatin_marks, n_features_per_mark)

    preds_per_feature = preds_per_feature.sum(axis=-1)  # sum over exponential basis function contributions
    preds_per_feature_proportion = preds_per_feature / preds_per_feature.sum(axis=-1, keepdims=True)

    # test = model.predict(xgb.DMatrix(alt_features)) - model.predict(xgb.DMatrix(ref_features))
    return preds_per_feature_proportion


def compute_effects(snpeffects, ref_preds, alt_preds, snpdists, snpstrands, all_models, maxshift=800, nfeatures=2002, batchSize=500):
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

    for i in range(int( (n_snps - 1) / batchSize) + 1):
        print("Processing " + str(i) + "th batch of "+str(batchSize))
        # compute gene expression change with models
        diff = reduce(lambda x, y: x + y, [np.tile(np.asarray(snpeffects[j][i * batchSize:(i + 1) * batchSize, :]), 10)
                                 * np.repeat(Xreducedall_diffs[j][i * batchSize:(i + 1) * batchSize, :], nfeatures, axis=1) for j in range(len(Xreducedall_diffs))])
        ref_features = reduce(lambda x, y: x + y, [np.tile(np.asarray(ref_preds[j][i * batchSize:(i + 1) * batchSize, :]), 10)
                                 * np.repeat(Xreducedall_diffs[j][i * batchSize:(i + 1) * batchSize, :], nfeatures, axis=1) for j in range(len(Xreducedall_diffs))])

        alt_features = reduce(lambda x, y: x + y, [np.tile(np.asarray(alt_preds[j][i * batchSize:(i + 1) * batchSize, :]), 10)
                                 * np.repeat(Xreducedall_diffs[j][i * batchSize:(i + 1) * batchSize, :], nfeatures, axis=1) for j in range(len(Xreducedall_diffs))])

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

    return effect, ref, alt, preds_per_feature_proportion


if __name__ == '__main__':
    main()
