cd /home/rshuai/research/ni-lab/ExPecto
source .env_expecto/bin/activate

# python3 interpret_features_grouped.py --inputFile ./resources/Xreducedall.2002.npy --belugaFeatures ./resources/deepsea_beluga_2002_features.tsv --expFile ./data/cultured/geneanno/geneanno.exp_cultured_primary_tubule_ENSEMBL.csv --targetIndex 1 --out_dir output_dir/interpret_features_grouped
python3 interpret_features_grouped.py --clustering_joblib output_dir/interpret_features_grouped/clustering_cached.joblib --inputFile ./resources/Xreducedall.2002.npy --belugaFeatures ./resources/deepsea_beluga_2002_features.tsv --expFile ./data/cultured/geneanno/geneanno.exp_cultured_primary_tubule_ENSEMBL.csv --targetIndex 1 --out_dir output_dir/interpret_features_grouped
