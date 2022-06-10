# -*- coding: utf-8 -*-
import argparse
import os
import subprocess

import pandas as pd
import pyfasta


def main():
    parser = argparse.ArgumentParser(description='Query FIMO for each SNP in SED df')
    parser.add_argument("--sed_file")
    parser.add_argument("--motif_file")
    parser.add_argument("--upstream_bp", default=30)
    parser.add_argument("--downstream_bp", default=30)
    parser.add_argument("--hg19_fasta", default="resources/hg19.fa")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_query_fimo_for_predictions', help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)

    # Read sed df
    sed_df = pd.read_csv(args.sed_file, sep="\t", index_col=0)
    sed_df["seq"] = sed_df.apply(lambda x: read_seq(x[0], x[1], x["strand"], x[3], x[4],
                                                    args.hg19_fasta, args.upstream_bp, args.downstream_bp), axis=1)

    fimo_in_fasta = f"{args.out_dir}/fimo_in.fasta"
    with open(fimo_in_fasta, "w") as f:
        for _, row in sed_df.iterrows():
            fasta_id = row[2]
            print(f'>{fasta_id}', file=f)
            print(row["seq"], file=f)

    fimo_out = f"{args.out_dir}/fimo_out.txt"
    with open(fimo_out, "w") as f:
        subprocess.call('fimo --thresh 1 --text {} {}'
                        .format(args.motif_file, fimo_in_fasta),
                        shell=True, stdout=f)


def read_seq(chrom, pos, strand, ref_snp, alt_snp, hg19_fasta, upstream_bp, downstream_bp):
    genome = pyfasta.Fasta(hg19_fasta)
    if strand == "+":
        start = pos - upstream_bp  # 1-based
        stop = pos + downstream_bp  # 1-based, inclusive
    elif strand == "-":
        start = pos - downstream_bp  # 1-based
        stop = pos + upstream_bp  # 1-based, inclusive

    seq = genome.sequence({'chr': chrom, 'start': start, 'stop': stop}).upper()

    if strand == "+":
        assert (seq[upstream_bp] == ref_snp) or (seq[upstream_bp] == alt_snp), "fasta does not match VCF"
    elif strand == "-":
        assert (seq[downstream_bp] == ref_snp) or (seq[downstream_bp] == alt_snp), "fasta does not match VCF"

    return seq

if __name__ == '__main__':
    main()
