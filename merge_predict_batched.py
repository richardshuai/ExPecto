# -*- coding: utf-8 -*-
import argparse
import glob
import os

import pandas as pd
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(description='Merge output batches of predict.py')
    parser.add_argument("--batch_dir", dest="batch_dir", type=str)
    parser.add_argument("--n_chunks", dest="n_chunks", type=int, default=1000, help="Expected number of chunks")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_cluster_by_pwm',
                        help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all sed files and save
    sed_files = natsorted(glob.glob(f"{args.batch_dir}/*/sed.tsv"))
    assert len(sed_files) == args.n_chunks, f"Expected {args.n_chunks} chunks but got {len(sed_files)} sed files"

    sed_df = pd.concat([pd.read_csv(sed_file, sep="\t") for sed_file in sed_files])
    sed_df.to_csv(f"{args.out_dir}/sed.tsv", sep="\t")


if __name__ == '__main__':
    main()
