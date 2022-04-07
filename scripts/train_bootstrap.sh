cd /home/rshuai/research/ni-lab/ExPecto
source .env_expecto/bin/activate

for i in {0..999}; do
	echo "Running for iteration $i"
	python3 train_bootstrap.py --inputFile ./resources/Xreducedall.2002.npy --expFile ./data/cultured/geneanno/geneanno.exp_cultured_primary_tubule_ENSEMBL.csv --targetIndex 1 --seed $i --output_dir output_dir/cultured_models/bootstrap_models/model_$i
done
