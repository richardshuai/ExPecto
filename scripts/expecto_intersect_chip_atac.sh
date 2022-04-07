cd /home/rshuai/research/ni-lab/ExPecto
source .env_expecto/bin/activate

echo "Computing features..."
python3 expecto_intersect_chip_atac.py resources/geneanno.csv /home/rshuai/research/ni-lab/analysis/kidney_data/primaryTubule_ATAC/HRCE1_peaks.narrowPeak --cuda --tf_only --windowsize 2000 -o intersect_expecto_tf_only

echo "Training with new computed features..."
python3 train.py --inputFile intersect_expecto_tf_only/Xreducedall.2002.atac_x_chip.npy --expFile ./data/cultured/geneanno/geneanno.exp_cultured_primary_tubule_ENSEMBL.csv --targetIndex 1 --output output_dir/cultured_models/cultured_primary_tubule_ENSEMBL_matched_with_basenji2_atac_x_chip_tf_only --plots_out_dir output_dir/cultured_models/plots/cultured_primary_tubule_ENSEMBL_matched_with_basenji2_atac_x_chip_tf_only --match_with_basenji2
