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


def main():
    parser = argparse.ArgumentParser(description='Compute SVD')
    parser.add_argument('replicate_expecto_features_dir')
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_svd',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # read in npy files
    npy_files = glob.glob(f'{args.replicate_expecto_features_dir}/*.npy')

    # tracks = np.empty((len(npy_files), 200, 2002), dtype=np.float32)
    # for i, npy_file in enumerate(tqdm(npy_files)):
    #     track = np.load(npy_file)
    #     tracks[i] = track

    tracks = np.empty((2002, len(npy_files), 200), dtype=np.float32)
    for i, npy_file in enumerate(tqdm(npy_files)):
        track = np.load(npy_file).T
        tracks[:, i] = track

    tracks = tracks.reshape((tracks.shape[0], -1))

    tf = tracks / tracks.sum(axis=-1, keepdims=True)  # term frequency
    idf = np.log(tracks.shape[0] / (1 + tracks.sum(axis=0)))  # inverse document freq (modified for continuous vals)

    del tracks
    tf_idf = tf * idf  # tf idf matrix

    # compute SVD
    svd = TruncatedSVD(n_components=100, random_state=1)
    svd.fit(tf_idf)

    dump(svd, f'{args.out_dir}/svd_100.joblib')


if __name__ == '__main__':
    main()
