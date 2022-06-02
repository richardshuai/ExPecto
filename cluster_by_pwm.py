# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd

from cluster_utils import get_keep_mask
import glob
from Bio import motifs

def main():
    parser = argparse.ArgumentParser(description='Cluster features by PWM')
    parser.add_argument('--belugaFeatures', action="store", dest="belugaFeatures",
                        help="tsv file denoting Beluga features")
    parser.add_argument('--jaspar_motif_db', action="store", dest="jaspar_motif_db",
                        help="JASPAR motif db path")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_cluster_by_pwm',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # setup
    np.random.seed(0)

    # Save clustering
    beluga_features_df = pd.read_csv(args.belugaFeatures, sep='\t', index_col=0)
    beluga_features_df['Assay type + assay + cell type'] = beluga_features_df['Assay type'] + '/' + beluga_features_df[
        'Assay'] + '/' + beluga_features_df['Cell type']

    # account for ablations
    keep_mask, hgnc_df = get_keep_mask(beluga_features_df, no_tf_features=False,
                              no_dnase_features=True, no_histone_features=True,
                              no_pol2=True, intersect_with_lambert=True, return_hgnc_df=True)
    hgnc_df = hgnc_df[keep_mask]

    # Load in JASPAR motifs
    motif_files = glob.glob(f"{args.jaspar_motif_db}/*.jaspar")
    motifs_in_db = []
    for motif_file in motif_files:
        with open(motif_file, "r") as f:
            motif_list = motifs.parse(f, "jaspar")
            assert len(motif_list) == 1, f"more than one motif found in {motif_file}"
            motifs_in_db.append(motif_list[0])

    included_motif_names = set(hgnc_df["Assay"].str.upper())
    motifs_found = set()
    cluster_motifs = []
    for motif in motifs_in_db:
        # Split motif heterodimers
        if len(motif.name.split('::')) > 1:
            print(motif.name)
            continue
        for tf_name in motif.name.split('::'):
            tf_name = tf_name.upper()
            if tf_name in included_motif_names:
                motifs_found.add(tf_name)
                cluster_motifs.append(motif)
                break

    print(f"Found {len(motifs_found)} motifs out of {len(included_motif_names)} motifs in hgnc df")
    with open(f"{args.out_dir}/cluster_motifs.jaspar", "w") as out_file:
        print(motifs.write(cluster_motifs, "jaspar"), file=out_file)
    print('end')




if __name__ == '__main__':
    main()
