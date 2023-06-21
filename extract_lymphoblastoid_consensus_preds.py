import argparse
import glob
import os
import multiprocessing as mp
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm
from functools import partial


def main():
    parser = argparse.ArgumentParser(description="Extract consensus predictions for Basenji and Expecto across both expression and chromatin for lymphoblastoid cell lines")
    parser.add_argument("--basenji_preds_dir", type=str, required=True, help="Directory containing basenji sed predictions for top eqtls")
    parser.add_argument("--targets_file", type=str, required=True, help="Basenji targets file")
    parser.add_argument("--n_center_bins", type=int, default=10, help="Number of bins to symmetrically average predictions over")
    parser.add_argument("--expecto_preds_dir", type=str, required=True, help="Directory containing expecto sed predictions for top eqtls")
    parser.add_argument("--beluga_features_tsv", type=str, required=True, help="TSV file containing beluga features")
    parser.add_argument('--eqtls_csv', type=str, required=True, help="CSV file containing top eqtls")
    parser.add_argument('--genes_csv', type=str, required=True, help="CSV file containing all genes")
    parser.add_argument("--extract_mode", type=str, default="snp", help="Extract predictions at SNP bin, TSS bin, or 50 bins symmetrically from TSS", choices=["snp", "tss", "50_bins"])
    parser.add_argument('--model', type=str, required=True, help="Model name to extract for", choices=["basenji", "expecto"])
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    # setup
    os.makedirs(args.out_dir, exist_ok=True)

    # read in feature files
    basenji_features_df = pd.read_csv(args.targets_file, sep="\t", index_col=0).reset_index(drop=True)
    expecto_features_df = pd.read_csv(args.beluga_features_tsv, sep="\t", index_col=0).reset_index(drop=True)

    # create ID column for both feature files
    expecto_features_df["ID"] = expecto_features_df.index.astype(str) + "|" + expecto_features_df["Cell type"] + "|" + expecto_features_df["Assay"] + "|" + expecto_features_df["Source"]
    basenji_features_df["ID"] = basenji_features_df.index.astype(str) + "|" + basenji_features_df["description"] + "|" + basenji_features_df["identifier"]

    # subset to GM12878
    basenji_gm12878 = basenji_features_df[basenji_features_df["description"].str.contains("GM12878")]
    expecto_gm12878 = expecto_features_df[expecto_features_df["Cell type"].str.contains("GM12878")]

    # read in eqtls
    eqtls_df = pd.read_csv(args.eqtls_csv, index_col=0).set_index("name")
    eqtls_df = eqtls_df

    # get strand
    genes_df = pd.read_csv(args.genes_csv, names=['ens_id', 'chrom', 'bp', 'gene_symbol', 'strand'], index_col=False)
    genes_df['name'] = genes_df['gene_symbol'].fillna(genes_df['ens_id']).str.lower()
    genes_df = genes_df.set_index('name')

    eqtls_df["strand"] = pd.merge(eqtls_df, genes_df, left_index=True, right_index=True, how="left")["strand"]
    assert set(eqtls_df["strand"]).issubset({"+", "-"}), f"Strand not found for all eqtls"

    # read in predictions for each eqtl
    if args.model == "basenji":
        # Basenji
        f = partial(process_eqtl_basenji, args=args, basenji_features_df=basenji_gm12878)
        with mp.Pool() as pool:
            if args.extract_mode == "snp":
                _ = list(tqdm(pool.imap_unordered(f, eqtls_df.iterrows()), total=eqtls_df.shape[0]))
            else:
                _ = list(tqdm(pool.imap_unordered(f, genes_df.iterrows()), total=genes_df.shape[0]))

    elif args.model == "expecto":
        # ExPecto
        df = genes_df if args.extract_mode != "snp" else eqtls_df
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            gene = row.name

            if args.extract_mode == "snp":
                # Extract preds at SNP per gene-SNP pair
                snp = row["SNP_ID"]
                preds_out_dir = f"{args.out_dir}/{gene}_{snp}"
            else:
                # Extract preds at TSS per gene
                preds_out_dir = f"{args.out_dir}/{gene}"
            Path(preds_out_dir).mkdir(parents=True, exist_ok=True)

            # read in expecto preds
            chromatin_file = f"{args.expecto_preds_dir}/{gene}/{gene}_chromatin.h5"
            with h5py.File(chromatin_file, "r") as gene_h5:
                expecto_preds = gene_h5["chromatin_preds"]
                sample_names = [x.decode("utf-8").split("|")[1] for x in gene_h5["record_ids"]]

                if args.extract_mode != "50_bins":
                    if args.extract_mode == "snp":
                        target_bin = get_snp_bin(row["SNPpos"], row["TSSpos_x"], row["strand"], model="expecto")
                    elif args.extract_mode == "tss":
                        target_bin = get_snp_bin(row["bp"], row["bp"], row["strand"], model="expecto")  # extract tss bin by setting snp pos to tss pos
                    expecto_gm12878_preds = expecto_preds[:, target_bin, np.array(expecto_gm12878.index)]
                    expecto_gene_df = pd.DataFrame(expecto_gm12878_preds, index=sample_names, columns=expecto_gm12878["ID"])
                    expecto_gene_df.to_csv(f"{preds_out_dir}/expecto_preds.csv")
                else:
                    target_bin = get_snp_bin(row["bp"], row["bp"], row["strand"], model="expecto")  # extract tss bin by setting snp pos to tss pos

                    # extract 50 bins symetrically from TSS
                    with h5py.File(f"{preds_out_dir}/gm12878_preds.h5", "w") as h5f:
                        expecto_gm12878_preds = expecto_preds[:, target_bin - 50:target_bin + 51, np.array(expecto_gm12878.index)].astype(np.float16)

                        # save the data
                        h5f.create_dataset("all_preds", data=expecto_gm12878_preds, compression="gzip", compression_opts=9)
                        # save the sample names
                        h5f.create_dataset("sample_names", data=np.array(sample_names, dtype="S"))
                        # save the feature labels
                        h5f.create_dataset("features", data=np.array(expecto_gm12878["ID"], dtype=h5py.special_dtype(vlen=str)))


def process_eqtl_basenji(index_row: tuple, args: argparse.Namespace, basenji_features_df: pd.DataFrame):
    _, row = index_row
    gene = row.name
    if args.extract_mode == "tss":
        # Extract preds at TSS per gene
        preds_out_dir = f"{args.out_dir}/{gene}"
        if os.path.exists(f"{preds_out_dir}/basenji_preds.csv"):
            return
    elif args.extract_mode == "snp":
        # Extract preds at SNP per gene-SNP pair
        snp = row["SNP_ID"]
        preds_out_dir = f"{args.out_dir}/{gene}_{snp}"
    else:
        # extract all bins
        preds_out_dir = f"{args.out_dir}/{gene}"

    Path(preds_out_dir).mkdir(parents=True, exist_ok=True)

    # read in basenji preds
    sample_files = glob.glob(f"{args.basenji_preds_dir}/{gene}/all_bins_per_sample/*.h5")
    sample_names = [Path(x).stem for x in sample_files]

    if args.extract_mode != "50_bins":
        # initialize pd dataframe of NaNs to store preds
        basenji_preds_df = pd.DataFrame(index=sample_names, columns=basenji_features_df["ID"])

        # read in predictions and get prediction at SNP bin for GM12878 features
        for sample_file in sample_files:
            with h5py.File(sample_file, "r") as gene_h5:
                basenji_preds = gene_h5["all_preds"]
                if args.extract_mode == "tss":
                    bin = get_snp_bin(snp_pos=row["bp"], tss_pos=row["bp"], strand=row["strand"], model="basenji")  # set snp_pos to tss_pos to get TSS preds
                else:
                    bin = get_snp_bin(snp_pos=row["SNPpos"], tss_pos=row["TSSpos_x"], strand=row["strand"], model="basenji")
                basenji_gm12878_preds = basenji_preds[bin, np.array(basenji_features_df.index)]

                # store preds
                basenji_preds_df.loc[Path(sample_file).stem, :] = basenji_gm12878_preds

        # save to CSV
        if basenji_preds_df.isna().any().any():
            print(f"WARNING: NaNs found in basenji_gene_df for {gene}")
        basenji_preds_df.to_csv(f"{preds_out_dir}/basenji_preds.csv")
    else:
        # Extract 50 bins symmetrically from TSS and save as h5
        bin = get_snp_bin(snp_pos=row["bp"], tss_pos=row["bp"], strand=row["strand"], model="basenji")  # set snp_pos to tss_pos to get TSS preds

        # initialize empty array to store preds
        num_samples = len(sample_files)
        num_features = len(basenji_features_df["ID"])
        all_preds = np.empty((num_samples, 101, num_features), dtype=np.float16)  # 101 bins centered at TSS

        for i, sample_file in enumerate(sample_files):
            with h5py.File(sample_file, "r") as gene_h5:
                basenji_preds = gene_h5["all_preds"]
                all_preds[i] = basenji_preds[bin - 50:bin + 51, np.array(basenji_features_df.index)]

        # Save the 3D array as an h5 file
        with h5py.File(f"{preds_out_dir}/gm12878_preds.h5", "w") as f:
            f.create_dataset("all_preds", data=all_preds, compression="gzip", compression_opts=9)
            f.create_dataset("sample_names", data=np.array(sample_names, dtype="S"))
            f.create_dataset("features", data=np.array(basenji_features_df.index))


def get_snp_bin(snp_pos: int, tss_pos: int, strand: str, model: str) -> int:
    """
    Calculate the bin for a given SNP position and strand.
    """
    if model == "expecto":
        windowsize = 2000
        shifts = np.array(list(range(-20000, 20000, 200)))

        if strand == "+":
            s = 1
        elif strand == "-":
            s = -1
        else:
            assert False, f"strand {strand} not recognized"

        snp_rel_pos = snp_pos - tss_pos
        tss_i = 0  # relative TSS position in the context sequence

        for i, shift in enumerate(shifts):
            bin_start = tss_i + (shift * s) - int(windowsize / 2 - 1)
            bin_end = tss_i + (shift * s) + int(windowsize / 2) + 1

            if bin_start <= snp_rel_pos < bin_end:
                return i

        assert False, f"SNP position {snp_pos} not found in any bin"

    if model == "basenji":
        seq_len = 131072
        bin_resolution = 128
        num_bins = 1024
        cropped_bins = 896

        # Calculate the TSS index in the input sequence
        if strand == "+":
            tss_index = seq_len // 2 - 1
        elif strand == "-":
            tss_index = seq_len // 2
        else:
            assert False, f"strand {strand} not recognized"

        # Calculate the relative SNP position to the TSS
        snp_rel_pos = snp_pos - tss_pos

        # Calculate the SNP index in the input sequence
        snp_index = tss_index + snp_rel_pos

        # Check if the SNP index is within the input sequence
        if snp_index < 0 or snp_index >= seq_len:
            assert False, f"SNP index {snp_index} is out of the input sequence"

        # Calculate the SNP bin index in the output (before cropping)
        snp_bin = snp_index // bin_resolution

        # Calculate the number of bins cropped from each side
        cropped_bins_each_side = (num_bins - cropped_bins) // 2

        # Adjust the SNP bin index for the cropped output
        snp_bin_cropped = snp_bin - cropped_bins_each_side

        # Check if the SNP bin index is within the cropped output
        if snp_bin_cropped < 0 or snp_bin_cropped >= cropped_bins:
            assert False, f"SNP bin index {snp_bin_cropped} is out of the cropped output"

        return int(snp_bin_cropped)

    else:
        assert False, f"model {model} not recognized"


if __name__ == "__main__":
    main()

