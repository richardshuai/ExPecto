cd /home/rshuai/research/ni-lab/ExPecto
source .env_expecto/bin/activate

EXP_FILE=resources/geneanno.exp_pbmc.csv

for i in {1..5}; do
	python3 train.py --expFile $EXP_FILE --targetIndex $i --output models/pbmc/idx_${i} --plots_out_dir models/pbmc/plots/idx_${i}
done
