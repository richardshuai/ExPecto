cd /home/rshuai/research/ni-lab/ExPecto || exit
source .env_expecto/bin/activate

# Run chromatin.py for chr1 to chr22
VCF_DIR=/home/rshuai/research/ni-lab/analysis/immune_atlas/ldsc/geno/1000G_EUR_Phase3_plink
BASE_OUT_DIR=/home/rshuai/research/ni-lab/ExPecto/output_dir/immune_atlas/chromatin

for i in {1..22}; do
  echo "Running chromatin.py for chr${i}..."
  vcf_file=$VCF_DIR/1000G.EUR.QC.${i}.vcf
  out_dir=$BASE_OUT_DIR/chr${i}
  python3 chromatin.py $vcf_file --cuda --batchsize 512 --output_dir $out_dir
done
