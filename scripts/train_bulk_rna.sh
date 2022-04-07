cd /home/rshuai/research/ni-lab/ExPecto
source .env_expecto/bin/activate

for exp_file in resources/bulk_rna_seq/geneannos/*; do
	identifier="${exp_file##*_}"  # get filename after last _
	identifier="${identifier%.*}"  # remove extension
	echo $identifier
	python3 train.py --expFile $exp_file --targetIndex 1 --output models/bulk_rna/${identifier} --plots_out_dir models/bulk_rna/plots/${identifier}
done
