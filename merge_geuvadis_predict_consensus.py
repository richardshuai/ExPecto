# -*- coding: utf-8 -*-
import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
import h5py
from natsort import natsorted
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Merge output batches of geuvadis_predict_consensus.py')
    parser.add_argument("--batch_dir", dest="batch_dir", type=str)
    parser.add_argument("--n_genes", dest="n_genes", type=int, default=3259, help="Expected number of genes")
    parser.add_argument('-o', dest="out_dir", type=str, default='merge_geuvadis_predict_consensus',
                        help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all sed files and save
    h5_files = natsorted(glob.glob(f"{args.batch_dir}/*/*.h5"))
    assert len(h5_files) == args.n_genes, f"Expected {args.n_genes} genes but got {len(h5_files)} h5 files"

    record_ids = None
    preds = []
    for h5_file in tqdm(h5_files):
        with h5py.File(h5_file, "r") as preds_h5:
            if record_ids is None:
                record_ids = np.array([parse_record_id(x) for x in preds_h5["record_ids"]])
            else:
                curr_record_ids = np.array([parse_record_id(x) for x in preds_h5["record_ids"]])
                assert (record_ids == curr_record_ids).all()
            preds.append(np.array(preds_h5["preds"]))

    preds = np.stack(preds)
    genes = [Path(x).stem for x in h5_files]
    with h5py.File(f"{args.out_dir}/expecto_preds.h5", "w") as h5_out:
        h5_out.create_dataset("record_ids", data=np.array(record_ids, 'S'))
        h5_out.create_dataset("genes", data=np.array(genes, 'S'))
        h5_out.create_dataset("preds", data=preds)


def parse_record_id(x):
    """
    Preprocess record ID to remove gene-specific info. e.g.
    b'chr19:58832097-58897632|NA20828|-|1pIu' -> NA20828|1pIu
    """
    x = x.decode("utf-8")
    x = x.split("|")
    return f"{x[1]}|{x[3]}"


if __name__ == '__main__':
    main()
