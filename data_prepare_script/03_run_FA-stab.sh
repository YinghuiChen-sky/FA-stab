#!/bin/bash
#SBATCH --job-name=e3nn       
#SBATCH --partition=gpu-a100-Partition 
#SBATCH --nodes=1               
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00          
#SBATCH --output=%j.out          
#SBATCH --error=%j.err           
source ~/.bashrc
conda init
conda activate /lustre/home/yhchen/anaconda3/envs/rna_esm
module load cuda/cuda-12.4

FAstab_dir="/lustre/home/yhchen/protein_stability/scripts/FA-stab/"

echo start `date`
/lustre/home/yhchen/anaconda3/envs/rna_esm/bin/python $FAstab_dir/inference_stability.py \
    --coords_seq_hdf5 ./wildtype_mutations_structures.h5 \
    --embed_label_h5 ./wildtype_mutations_seq_embedding.h5 \
    --testset_name Test \
    --mutations_csv 00_mutations_info_examples.csv \
    --output_dir ./ \
    --Stability_checkpoint $FAstab_dir/FA-stab_v2.pth \
    --gpu_model 0

echo end `date`