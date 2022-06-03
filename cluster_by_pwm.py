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
    parser.add_argument('--hocomoco_jaspar_motif_file', action="store", dest="hocomoco_jaspar_motif_file",
                        help="HOCOMOCO motifs file in JASPAR format")
    parser.add_argument('--cisbp2_meme_file', action="store", dest="cisbp2_meme_file",
                        help="CIS-BP 2 Homo sapiens meme file")
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
    jaspar_motif_files = glob.glob(f"{args.jaspar_motif_db}/*.jaspar")
    jaspar_motifs = []
    for motif_file in jaspar_motif_files:
        with open(motif_file, "r") as f:
            motif_list = motifs.parse(f, "jaspar")
            assert len(motif_list) == 1, f"more than one motif found in {motif_file}"
            jaspar_motifs.append(motif_list[0])

    # For each JASPAR motif, add if in beluga/lambert features
    included_motif_names = set(hgnc_df["Assay"].str.upper())
    motifs_found = set()
    cluster_motifs = []
    for motif in jaspar_motifs:
        # Split motif heterodimers
        if len(motif.name.split('::')) > 1:
            continue
        tf_name = motif.name.upper()
        if tf_name in included_motif_names:
            motifs_found.add(tf_name)
            cluster_motifs.append(motif)

    # Load in HOCOMOCO motifs
    with open(args.hocomoco_jaspar_motif_file, "r") as f:
        hocomoco_motifs = motifs.parse(f, "jaspar")

    for motif in hocomoco_motifs:
        tf_name = motif.name.split("_")[0].upper()
        if tf_name in included_motif_names:
            motifs_found.add(tf_name)
            cluster_motifs.append(motif)

    # TODO: Not including CIS-BP motifs for now. Noticing that many are binary PWMs...
    # # Load meme motifs
    # with open(args.cisbp2_meme_file, "r") as f:
    #     cisbp2_motifs = motifs.parse(f, "MINIMAL")
    #
    # # Preprocess CIS-BP meme format
    # motif_id_to_name = {}
    # with open(args.cisbp2_meme_file, "r") as f:
    #     for line in f.readlines():
    #         if line.startswith("MOTIF"):
    #             _, motif_id, motif_name = line.strip().split(" ")
    #             # deal with motif names such as '(CDC5L)_(Arabidopsis_thaliana)_(DBD_0.87)'
    #             if motif_name.startswith("("):
    #                 motif_name = motif_name.split("_")[0][1:-1]
    #             motif_name = motif_name.upper()
    #             motif_id_to_name[motif_id] = motif_name
    #
    # for motif in cisbp2_motifs:
    #     motif.matrix_id = motif.name
    #     motif.base_id = motif.name
    #     motif.name = motif_id_to_name[motif.base_id]
    #     tf_name = motif.name
    #     if tf_name in included_motif_names:
    #         motifs_found.add(tf_name)
    #         cluster_motifs.append(motif)

    print(f"Found {len(motifs_found)} motifs out of {len(included_motif_names)} motifs in hgnc df")
    with open(f"{args.out_dir}/cluster_motifs.jaspar", "w") as out_file:
        print(motifs.write(cluster_motifs, "jaspar"), file=out_file)


if __name__ == '__main__':
    main()
