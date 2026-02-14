#!/bin/bash
#SBATCH --job-name=inputdata       
#SBATCH --partition=gpu-a100-Partition 
#SBATCH --nodes=1               
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00          
#SBATCH --output=%j.out          
#SBATCH --error=%j.err           

source ~/.bashrc
conda init
conda activate /lustre/home/yhchen/anaconda3/envs/rna_esm
module load cuda/cuda-12.4

mut_csv="00_mutations_info_examples.csv"
AF3_dir="00_wild_type_AF3_pred"
esm_model_dir="/lustre/home/yhchen/software/Evolutionary_Scale_Modeling"
FAstab_dir="/lustre/home/yhchen/protein_stability/scripts/FA-stab/"

MSA_AF3="./mutations_embedding/MSA_AF3"
mkdir -p $MSA_AF3

####### extract MSA from AF3 json files and filter them
/lustre/home/yhchen/anaconda3/envs/rna_esm/bin/python $FAstab_dir/data_prepare_script/01_AF3_MSA_extract.py \
                            --AF3_json_dir $AF3_dir \
                            --out_msa_dir  $MSA_AF3 \
                            --mutations_sample_csv $mut_csv


####### get ESM embeddings  for wildtype and mutated proteins
/lustre/home/yhchen/anaconda3/envs/rna_esm/bin/python $FAstab_dir/data_prepare_script/02_Mutations_ESM_embed_label.py --esm_model_dir $esm_model_dir \
                            --out_hdf5_path wildtype_mutations_seq_embedding.h5 \
                             --filtered_msa_dir  $MSA_AF3 \
                             --ddG_train_data_csv $mut_csv \
                            --device "cuda:0"