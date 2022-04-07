cd /home/rshuai/research/ni-lab/ExPecto
source .env_expecto/bin/activate

echo "Computing features..."
python3 compute_expecto_features.py resources/geneanno.csv /home/rshuai/research/ni-lab/analysis/basenji2/tss/cultured_primary_tubule/representative_tss_top/tss.tsv --cuda --windowsize 2000 -o compute_expecto_features_no_default

echo "Training with new computed features..."
python3 train.py --inputFile compute_expecto_features_no_default/Xreducedall.2002.representative_tss_top.npy --expFile ./data/cultured/geneanno/geneanno.exp_cultured_primary_tubule_ENSEMBL.csv --targetIndex 1 --output output_dir/cultured_models/cultured_primary_tubule_ENSEMBL_matched_with_basenji2_representative_tss_top_no_default --plots_out_dir output_dir/cultured_models/plots/cultured_primary_tubule_ENSEMBL_matched_with_basenji2_representative_tss_top_no_default --match_with_basenji2
