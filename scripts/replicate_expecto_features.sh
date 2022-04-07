cd /home/rshuai/research/ni-lab/ExPecto
source .env_expecto/bin/activate

echo "Computing features..."
python3 replicate_expecto_features.py resources/geneanno.csv --cuda --windowsize 2000 -o output_dir/replicate_expecto_features_all
