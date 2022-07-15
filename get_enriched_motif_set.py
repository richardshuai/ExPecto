# -*- coding: utf-8 -*-
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Query FIMO for each SNP in SED df')
    parser.add_argument("--hypergeom_enrichment_tsv")
    parser.add_argument("--motif_db_file")
    parser.add_argument("--qval_thresh", default=0.01)
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_enriched_motif_set', help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)

    # Read in hypergeometric TSV and filter by qval
    enrichment_df = pd.read_csv(args.hypergeom_enrichment_tsv, sep="\t", index_col=0)
    motif_set = set(enrichment_df[enrichment_df["hypergeom_qval"] < 0.01].index)

    with open(args.motif_db_file, "r") as f:
        lines = f.readlines()

    motifs_found = set()
    with open(f"{args.out_dir}/enriched_motifs.meme", "w") as out_file:
        write_mode = True
        for line in lines:
            line = line.strip()
            if line[:5] != "MOTIF" and write_mode:
                print(line, file=out_file)
            elif line[:5] == "MOTIF":
                write_mode = False
                _, motif_id, motif_name = line.split()
                if motif_id in motif_set:
                    motifs_found.add(motif_id)
                    write_mode = True
                    print(line, file=out_file)

    assert motif_set == motifs_found, "Did not find all motifs in enriched motif set in the motif db file"


if __name__ == '__main__':
    main()
