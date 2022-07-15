# -*- coding: utf-8 -*-
import argparse
import os
import subprocess

import pandas as pd
import pyfasta


def main():
    parser = argparse.ArgumentParser(description='Query FIMO for each SNP in SED df')
    parser.add_argument("--vcf_file")
    parser.add_argument("--motif_file")
    parser.add_argument("--chunk_size", action="store", dest="chunk_size", type=int, default=None,
                        help="Size of chunks for batching predictions")
    parser.add_argument("--chunk_i", action="store", dest="chunk_i", type=int, default=None,
                        help="Chunk index for current run, starting from 0")
    parser.add_argument("--bp_pad", default=30)
    parser.add_argument("--pval_match_threshold", default=1e-4)
    parser.add_argument("--hg19_fasta", default="resources/hg19.fa")
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_query_fimo_for_predictions', help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)

    # Read vcf df
    vcf_df = pd.read_csv(args.vcf_file, sep="\t", comment="#",
                         names=["CHROM", "POS", "ID", "REF", "ALT" , "QUAL", "FILTER", "INFO"])

    if args.chunk_i is not None:
        vcf_df = vcf_df.iloc[args.chunk_i * args.chunk_size:(args.chunk_i + 1) * args.chunk_size]

    vcf_df["seq"] = vcf_df.apply(lambda x: read_seq(x["CHROM"], x["POS"], x["REF"], x["ALT"],
                                                    args.hg19_fasta, args.bp_pad), axis=1)

    fimo_in_fasta = f"{args.out_dir}/fimo_in.fasta"
    with open(fimo_in_fasta, "w") as f:
        for _, row in vcf_df.iterrows():
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
    fimo_df = fimo_df[(fimo_df["start"] <= (args.bp_pad + 1)) & (fimo_df["stop"] >= (args.bp_pad + 1))]

    # get most significant match for each motif-variant pair
    fimo_df = fimo_df.sort_values(by="p-value").drop_duplicates(subset=["motif_id", "motif_alt_id", "sequence_name"], keep="first")

    fimo_df.to_csv(f"{args.out_dir}/fimo_filtered.tsv", sep="\t", header=True)


def read_seq(chrom, pos, ref_snp, alt_snp, hg19_fasta, bp_pad):
    genome = pyfasta.Fasta(hg19_fasta)
    start = pos - bp_pad  # 1-based
    stop = pos + bp_pad  # 1-based, inclusive

    seq = genome.sequence({'chr': chrom, 'start': start, 'stop': stop}).upper()

    assert (seq[bp_pad:bp_pad + len(ref_snp)] == ref_snp) or (seq[bp_pad:bp_pad + len(alt_snp)] == alt_snp), "fasta does not match VCF"
    return seq


if __name__ == '__main__':
    main()
