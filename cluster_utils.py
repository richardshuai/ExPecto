import numpy as np


def get_keep_mask(args, beluga_features_df):
    keep_mask = np.ones(beluga_features_df.shape[0], dtype=bool)

    if args.no_tf_features:
        print("not including TF features")
        keep_mask = keep_mask & (beluga_features_df['Assay type'] != 'TF')

    if args.no_dnase_features:
        print("not including DNase features")
        keep_mask = keep_mask & (beluga_features_df['Assay type'] != 'DNase')

    if args.no_histone_features:
        print("not including histone features")
        keep_mask = keep_mask & (beluga_features_df['Assay type'] != 'Histone')

    if args.intersect_with_lambert:
        print("intersecting with Lambert data")
        lambert_df = pd.read_csv('./resources/Lambert2018_TFs_v_1.01_curatedTFs.csv', index_col=0)
        keep_mask = keep_mask & (beluga_features_df['Assay'].isin(lambert_df['HGNC symbol']))

    if args.no_pol2:
        print("taking out Pol2*")
        keep_mask = keep_mask & ~(beluga_features_df['Assay'].str.startswith('Pol'))

    return keep_mask