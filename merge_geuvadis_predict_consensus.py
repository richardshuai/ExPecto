# -*- coding: utf-8 -*-
import argparse
import glob
import os
import numpy as np
from tqdm import tqdm
import h5py
from natsort import natsorted


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
    for h5_file in tqdm(h5_files[:10]):
        with h5py.File(h5_file, "r") as preds_h5:
            if record_ids is None:
                record_ids = np.array(preds_h5["record_ids"])
            else:
                import pdb; pdb.set_trace()
                assert (record_ids == np.array(preds_h5["record_ids"])).all()
            preds.append(np.array(preds_h5["preds"]))
    
    preds = np.concatenate(preds)
    with open(f"{args.out_dir}/expecto_preds.h5", "w") as h5_out:
        h5_out.create_dataset("record_ids", data=record_ids)
        h5_out.create_dataset("preds", data=preds)


if __name__ == '__main__':
    main()
