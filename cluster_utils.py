import numpy as np
import pandas as pd

# LAMBERT_CSV_PATH = './resources/Lambert2018_TFs_v_1.01_curatedTFs.csv'
LAMBERT_HGNC_PATH = './resources/Lambert-hgnc-symbol-check.csv'
HGNC_MAPPING_PATH = './resources/beluga_hgnc_mapping.csv'

def get_keep_mask(beluga_features_df, no_tf_features, no_dnase_features,
                  no_histone_features, intersect_with_lambert, no_pol2, return_hgnc_df=False):
    hgnc_df = None
    keep_mask = np.ones(beluga_features_df.shape[0], dtype=bool)

    if no_tf_features:
        print("not including TF features")
        keep_mask = keep_mask & (beluga_features_df['Assay type'] != 'TF')

    if no_dnase_features:
        print("not including DNase features")
        keep_mask = keep_mask & (beluga_features_df['Assay type'] != 'DNase')

    if no_histone_features:
        print("not including histone features")
        keep_mask = keep_mask & (beluga_features_df['Assay type'] != 'Histone')

    if intersect_with_lambert:
        print("intersecting with Lambert data")
        lambert_df = pd.read_csv(LAMBERT_HGNC_PATH, index_col=0)
        beluga_hgnc_mapping = pd.read_csv(HGNC_MAPPING_PATH, index_col=0).dropna(subset=["Approved symbol"])
        hgnc_assays = list(beluga_features_df['Assay'].values)
        for i, assay in enumerate(hgnc_assays):
            if assay in beluga_hgnc_mapping.index:
                match = beluga_hgnc_mapping.loc[assay][["Match type", "Approved symbol"]]
                if len(match.shape) != 1:
                    match = match[match["Match type"] == "Approved symbol"].iloc[0]
                hgnc_assays[i] = match["Approved symbol"].upper()

        hgnc_df = beluga_features_df.copy()
        hgnc_df["Assay"] = hgnc_assays

        keep_mask = keep_mask & (hgnc_df['Assay'].isin(lambert_df['Approved symbol'].values))
        keep_mask = keep_mask & (~hgnc_df['Assay'].isnull())

    if no_pol2:
        print("taking out Pol2*")
        keep_mask = keep_mask & ~(beluga_features_df['Assay'].str.startswith('Pol'))

    if return_hgnc_df:
        return keep_mask, hgnc_df
    return keep_mask