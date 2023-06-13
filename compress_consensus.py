import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Compress basenji consensus preds by reducing precision and compressing with gzip")
    parser.add_argument("--basenji_consensus_preds_dir", type=str, required=True, help="Directory containing basenji consensus predictions")
    parser.add_argument("--num_chunks", dest="num_chunks", default=None, type=int, help="Number of chunks to split computation into")
    parser.add_argument("--chunk_i", dest="chunk_i", default=None, type=int, help="chunk index (0-indexed)")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    # setup
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # load in h5 file predictions for center bins
    center_h5_files = natsorted(glob.glob(f"{args.basenji_consensus_preds_dir}/*/*.h5"))

    # split into chunks
    if args.num_chunks is not None:
        chunks = np.array_split(center_h5_files, args.num_chunks)
        center_h5_files = chunks[args.chunk_i]

    # save reduced precision predictions
    print("Reducing precision for center bin h5 files...")
    for center_h5_file in tqdm(center_h5_files):
        center_out_dir = f"{args.out_dir}/{Path(center_h5_file).parent.name}"
        Path(center_out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(center_h5_file, "r") as f:
            preds = f["preds"][...]
            preds = preds.astype(np.float16)
            out_file = f"{center_out_dir}/{Path(center_h5_file).name}"
            with h5py.File(out_file, "w") as f_out:
                f_out.create_dataset("preds", data=preds, compression="gzip", compression_opts=9)

        # delete original file
        os.remove(center_h5_file)

    # load in h5 file predictions for all bins for each sample
    sample_h5_files = natsorted(glob.glob(f"{args.basenji_consensus_preds_dir}/*/all_bins_per_sample/*.h5"))

    # split into chunks
    if args.num_chunks is not None:
        chunks = np.array_split(sample_h5_files, args.num_chunks)
        sample_h5_files = chunks[args.chunk_i]

    # save reduced precision predictions
    print("Reducing precision for all bin h5 files...")
    for sample_h5_file in tqdm(sample_h5_files):
        sample_out_dir = f"{args.out_dir}/{Path(sample_h5_file).parent.parent.name}/all_bins_per_sample"
        Path(sample_out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(sample_h5_file, "r") as f:
            preds = f["preds"][...]
            preds = preds.astype(np.float16)
            out_file = f"{sample_out_dir}/{Path(sample_h5_file).name}"
            with h5py.File(out_file, "w") as f_out:
                f_out.create_dataset("preds", data=preds, compression="gzip", compression_opts=9)

        # delete original file
        os.remove(sample_h5_file)


if __name__ == "__main__":
    main()

