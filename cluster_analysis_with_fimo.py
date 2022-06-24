# -*- coding: utf-8 -*-
import argparse
import os

import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict
import matplotlib.pyplot as plt
from util.rank_based_inverse_normal_transformation import rank_INT


def main():
    parser = argparse.ArgumentParser(description='Analyze fimo results with cluster contribs')
    parser.add_argument("--cluster_contribs_file")
    parser.add_argument("--rsat_clusters_file")
    parser.add_argument("--fimo_out_file")
    parser.add_argument("--rank_int", default=False, action="store_true",
                        help="use rank-based inverse normal transformation for SED scores")
    parser.add_argument("--upstream_bp", default=30)
    parser.add_argument("--downstream_bp", default=30)
    parser.add_argument("--pval_match_threshold", default=1e-4)
    parser.add_argument('-o', dest="out_dir", type=str, default='temp_cluster_analysis_with_fimo', help='Output directory')
    args = parser.parse_args()

    # Setup
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(0)

    # Read cluster contribs df and fimo out
    rsat_clusters_df = pd.read_csv(args.rsat_clusters_file, sep="\t", header=None, index_col=0)
    cluster_contribs_df = pd.read_csv(args.cluster_contribs_file, sep="\t", index_col=0).drop("cluster_-1", axis=1) \
                                                                                        .reset_index(drop=True)

    if args.rank_int:
        # apply rank inverse normal transform to SED scores
        cluster_contribs_df.insert(
            loc=cluster_contribs_df.columns.tolist().index("SED") + 1,
            column="SED_RINT",
            value=cluster_contribs_df.groupby("gene")["SED"].transform(lambda x: rank_INT(x, stochastic=True))
        )
        cluster_contribs_df = cluster_contribs_df.drop("SED", axis=1)

    column_names = ['motif_id', 'motif_alt_id', 'sequence_name', 'start',
                    'stop', 'strand', 'score', 'p-value', 'q-value', 'matched_sequence']
    fimo_df = pd.read_table(args.fimo_out_file, sep='\t', names=column_names, comment='#')

    n_motifs_total = len(set(fimo_df["motif_alt_id"]))
    assert n_motifs_total == len(set(sum(rsat_clusters_df[1][:-1].str.split(",").tolist(), [])))  # skip cluster_-1

    # subset fimo df to motifs within range of variant
    fimo_df = fimo_df[(fimo_df["start"] <= (args.upstream_bp + 1)) & (fimo_df["stop"] >= (args.upstream_bp + 1))]

    # get most significant match for all motifs for all variants
    fimo_df = fimo_df.sort_values(by="p-value").drop_duplicates(subset=["motif_id", "motif_alt_id", "sequence_name"], keep="first")

    # filter by min pval threshold
    fimo_df = fimo_df[fimo_df["p-value"] < args.pval_match_threshold]

    # for each variant in cluster_contribs_df, get proportion of matches in top clusters vs. non-top clusters
    hypergeom_df, n_cluster_to_unique_clusters = cluster_contribs_hypergeom(cluster_contribs_df, fimo_df, rsat_clusters_df,
                                                                            return_unique_clusters=True)

    n_unique_clusters_df = pd.DataFrame.from_dict({k: len(v) for k, v in n_cluster_to_unique_clusters.items()},
                                                  orient="index", columns=["n_unique_clusters"])
    plt.plot(n_unique_clusters_df.index, n_unique_clusters_df["n_unique_clusters"])
    plt.xlabel("Number of top clusters")
    plt.ylabel("Number of unique clusters")
    plt.title("Number of unique top clusters across all variants")
    plt.ylim(0, n_unique_clusters_df["n_unique_clusters"].max() + 1)
    plt.savefig(f"{args.out_dir}/num_unique_clusters.pdf", dpi=300)
    plt.show()

    # Shuffled clusters
    shuffled_contribs_df = cluster_contribs_df.copy()
    shuffled_contribs_df.iloc[:, 15:] = shuffle_along_axis(shuffled_contribs_df.iloc[:, 15:].values, axis=1)
    shuffled_contribs_hypergeom_df = cluster_contribs_hypergeom(shuffled_contribs_df,
                                                                fimo_df, rsat_clusters_df)

    # Shuffled variants
    shuffled_variants_df = cluster_contribs_df.copy()
    random_idxs = np.random.choice(cluster_contribs_df.shape[0], cluster_contribs_df.shape[0], replace=False)
    shuffled_variants_df.loc[:, "2"] = shuffled_variants_df.loc[random_idxs, "2"].reset_index(drop=True)

    shuffled_variant_hypergeom_df = cluster_contribs_hypergeom(shuffled_variants_df,
                                                               fimo_df, rsat_clusters_df)

    # By SED proportion percentile
    percentile_hypergeom_dfs = {}
    for percentile_range in [(x, x + 25) for x in range(0, 100, 25)]:
        if args.rank_int:
            sed_column = "SED_RINT"
        else:
            sed_column = "SED_PROPORTION"
        lower, upper = [np.percentile(cluster_contribs_df[sed_column], p) for p in percentile_range]
        percentile_contribs_df = cluster_contribs_df[(lower <= cluster_contribs_df[sed_column]) & \
                                                   (cluster_contribs_df[sed_column] <= upper)]
        percentile_hypergeom_df = cluster_contribs_hypergeom(percentile_contribs_df, fimo_df, rsat_clusters_df)
        percentile_hypergeom_dfs[percentile_range] = percentile_hypergeom_df

    # Plots
    plt.plot(hypergeom_df["top_cluster_idx"], -np.log10(hypergeom_df["hypergeom_pval"]), label="top")
    plt.plot(shuffled_contribs_hypergeom_df["top_cluster_idx"], -np.log10(shuffled_contribs_hypergeom_df["hypergeom_pval"]), label="shuffled clusters")
    plt.plot(shuffled_variant_hypergeom_df["top_cluster_idx"], -np.log10(shuffled_variant_hypergeom_df["hypergeom_pval"]), label="shuffled variants")
    plt.xlabel("Top cluster index")
    plt.ylabel(f'-$\log_{{10}}$ pval')
    plt.title("Hypergeometric pval vs. number of clusters included")
    plt.legend()
    plt.savefig(f"{args.out_dir}/hypergeom_test_vs_cluster.pdf", dpi=300)
    plt.show()

    # Plot by quantile
    for percentile_range, percentile_hypergeom_df in percentile_hypergeom_dfs.items():
        plt.plot(percentile_hypergeom_df["top_cluster_idx"],
                 -np.log10(percentile_hypergeom_df["hypergeom_pval"]), label=f"percentile: {percentile_range}",
                 ls="--")
    plt.xlabel("Top cluster index")
    plt.ylabel(f'-$\log_{{10}}$ pval')
    plt.title("Hypergeometric pval vs. top cluster index")
    plt.legend()
    plt.savefig(f"{args.out_dir}/hypergeom_test_vs_cluster_by_quantile.pdf", dpi=300)
    plt.show()
    print("end")


def cluster_contribs_hypergeom(cluster_contribs_df, fimo_df, rsat_clusters_df, n_neg_clusters=20,
                               return_unique_clusters=False):
    hypergeom_data = {"top_cluster_idx": [], "hypergeom_pval": []}
    if return_unique_clusters:
        n_cluster_to_unique_clusters = defaultdict(set)

    for top_cluster_idx in range(rsat_clusters_df.shape[0] - n_neg_clusters):
        n_pos_matches = 0
        n_pos_motifs = 0
        n_neg_matches = 0
        n_neg_motifs = 0
        for _, row in cluster_contribs_df.iterrows():
            rsid = row[2]
            cluster_contribs = row.iloc[15:].sort_values(ascending=False, key=np.abs)
            cluster_i = cluster_contribs.index[top_cluster_idx]
            if return_unique_clusters:
                n_cluster_to_unique_clusters[top_cluster_idx] = \
                    n_cluster_to_unique_clusters[top_cluster_idx].union(set(cluster_contribs.index[:top_cluster_idx + 1].tolist()))


            # Get matches for this variant
            rsid_fimo_df = fimo_df[fimo_df["sequence_name"] == rsid]

            # Number of motifs with matches vs. number of motifs queried for cluster i
            pos_motifs = set(sum([x.split(",") for x in rsat_clusters_df.loc[cluster_i]], []))
            cluster_matches_df = rsid_fimo_df[rsid_fimo_df["motif_alt_id"].isin(pos_motifs)]
            n_pos_matches += cluster_matches_df.shape[0]
            n_pos_motifs += len(pos_motifs)

            # Number of motifs with matches vs. number of motifs queried for bottom clusters
            bottom_clusters = cluster_contribs.index[-n_neg_clusters:]
            neg_motifs = set(sum([x.split(",") for x in rsat_clusters_df.loc[bottom_clusters].squeeze(1)], []))
            other_matches_df = rsid_fimo_df[rsid_fimo_df["motif_alt_id"].isin(neg_motifs)]
            n_neg_matches += other_matches_df.shape[0]
            n_neg_motifs += len(neg_motifs)

        k, M, n, N = n_pos_matches, n_pos_motifs + n_neg_motifs, n_pos_motifs, n_pos_matches + n_neg_matches
        hypergeom_pval = hypergeom.sf(k - 1, M, n, N)
        print(f"Top cluster index: {top_cluster_idx}, hypergeometric pval: {hypergeom_pval}")
        hypergeom_data["top_cluster_idx"].append(top_cluster_idx)
        hypergeom_data["hypergeom_pval"].append(hypergeom_pval)

    hypergeom_df = pd.DataFrame(hypergeom_data)
    if return_unique_clusters:
        return hypergeom_df, n_cluster_to_unique_clusters
    return hypergeom_df


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


if __name__ == '__main__':
    main()
