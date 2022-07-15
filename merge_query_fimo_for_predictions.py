# -*- coding: utf-8 -*-
import argparse
import glob
import os

import pandas as pd
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(description='Merge output batches of query_fimo_for_predictions.py')
    parser.add_argument("--batch_dir", dest="batch_dir", type=str)
    parser.add_argument("--n_chunks", dest="n_chunks", type=int, default=25, help="Expected number of chunks")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_merge_query_fimo_for_predictions',
                        help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all fimo filtered files and save
    fimo_filtered_files = natsorted(glob.glob(f"{args.batch_dir}/*/fimo_filtered.tsv"))
    assert len(fimo_filtered_files) == args.n_chunks, f"Expected {args.n_chunks} chunks but got {len(fimo_filtered_files)} fimo filtered files"

    fimo_df = pd.concat([pd.read_csv(fimo_filtered_file, sep="\t") for fimo_filtered_file in fimo_filtered_files])
    fimo_df = fimo_df.sort_values(by="p-value").drop_duplicates(subset=["motif_id", "motif_alt_id", "sequence_name"],
                                                                keep="first")
    fimo_df.to_csv(f"{args.out_dir}/fimo_filtered.tsv", sep="\t", header=True)


if __name__ == '__main__':
    main()
