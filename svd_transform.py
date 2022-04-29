# -*- coding: utf-8 -*-
import argparse
import math
import pyfasta
import torch
from torch import nn
import numpy as np
import pandas as pd
import h5py
import os
from tqdm import tqdm
import glob
from sklearn.decomposition import TruncatedSVD
from joblib import dump, load

from cluster_utils import get_keep_mask


def main():
    parser = argparse.ArgumentParser(description='Compute SVD transform of training data')
    parser.add_argument('replicate_expecto_features_dir')
    parser.add_argument('svd_joblib')
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
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_svd_transform',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # read in npy files
    npy_files = glob.glob(f'{args.replicate_expecto_features_dir}/*.npy')

    tracks = np.empty((2002, len(npy_files), 200), dtype=np.float32)
    for i, npy_file in enumerate(tqdm(npy_files)):
        track = np.load(npy_file).T
        tracks[:, i] = track

    tracks = tracks.reshape((tracks.shape[0], -1))

    # Ablations
    beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
    beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df[
        'Assay'] + '/' + beluga_features_df['Cell type']

    keep_mask = get_keep_mask(args, beluga_features_df)

    keep_indices = np.nonzero(keep_mask.values)[0]
    tracks = tracks[keep_indices]

    tracks = tracks.reshape((tracks.shape[0], -1))
    print(f'Tracks shape: {tracks.shape}')

    # tf-idf transform
    tf = tracks / tracks.sum(axis=-1, keepdims=True)  # term frequency
    idf = np.log(tracks.shape[0] / (1 + tracks.sum(axis=0)))  # inverse document freq (modified for continuous vals)

    del tracks
    tf_idf = tf * idf  # tf idf matrix

    # load SVD in
    svd = load(args.svd_joblib)

    # reduce dimensionality and save
    X = svd.transform(tf_idf)
    np.save(f'{args.out_dir}/tf_idf_reduced_100', X)


if __name__ == '__main__':
    main()
