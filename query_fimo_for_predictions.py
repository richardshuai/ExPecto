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
    parser.add_argument("--chunk_size", action="store", dest="chunk_size", type=int, default=None,
                        help="Size of chunks for batching predictions")
    parser.add_argument("--chunk_i", action="store", dest="chunk_i", type=int, default=None,
                        help="Chunk index for current run, starting from 0")
    parser.add_argument("--upstream_bp", default=30)
    parser.add_argument("--downstream_bp", default=30)
    parser.add_argument("--pval_match_threshold", default=1e-4)
    parser.add_argument("--hg19_fasta", default="resources/hg19.fa")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_query_fimo_for_predictions', help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)

    # Read sed df
    sed_df = pd.read_csv(args.sed_file, sep="\t").iloc[:, 2:]

    if args.chunk_i is not None:
        sed_df = sed_df.iloc[args.chunk_i * args.chunk_size:(args.chunk_i + 1) * args.chunk_size]

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

    column_names = ['motif_id', 'motif_alt_id', 'sequence_name', 'start',
                    'stop', 'strand', 'score', 'p-value', 'q-value', 'matched_sequence']
    fimo_df = pd.read_table(fimo_out, sep='\t', names=column_names, comment='#')

    # subset fimo df to queries within range of variant
    fimo_df = fimo_df[(fimo_df["start"] <= (args.upstream_bp + 1)) & (fimo_df["stop"] >= (args.upstream_bp + 1))]

    # get most significant match for each motif-variant pair
    fimo_df = fimo_df.sort_values(by="p-value").drop_duplicates(subset=["motif_id", "motif_alt_id", "sequence_name"], keep="first")

    fimo_df.to_csv(f"{args.out_dir}/fimo_filtered.tsv", sep="\t", header=True)


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
