#!/bin/bash
#SBATCH --job-name=inputdata       
#SBATCH --partition=cu-1
#SBATCH --nodes=1               
#SBATCH --mem=48G
#SBATCH --ntasks=4
#SBATCH --time=100:00:00          
#SBATCH --output=%j.out         
#SBATCH --error=%j.err           

source ~/.bashrc
conda init
conda activate /lustre/home/yhchen/anaconda3/envs/rna_esm
module load cuda/cuda-12.4

mut_csv="00_mutations_info_examples.csv"
AF3_dir="00_wild_type_AF3_pred"
foldx5="/lustre/home/yhchen/software/FoldX/FoldX5/FoldX5_2026/foldx5"
FAstab_dir="/lustre/home/yhchen/protein_stability/scripts/FA-stab/"

####### transfer AF3 cif files to pdb files and repair them
mkdir -p ./mutations_structure/wildtype_AF3_preds  ./mutations_structure/wildtype_AF3_preds_repair

/lustre/home/yhchen/anaconda3/envs/rna_esm/bin/python $FAstab_dir/data_prepare_script/01_CIF2PDB.py --AF3_dir  $AF3_dir \
                  --out_pdb  ./mutations_structure/wildtype_AF3_preds --sample_csv $mut_csv \
                  --foldx5 $foldx5 \
                  --out_repair_dir ./mutations_structure/wildtype_AF3_preds_repair

####### build structures of mutated proteins by FoldX in parallel
mkdir -p ./mutations_structure/mutations_FoldX


/lustre/home/yhchen/anaconda3/envs/rna_esm/bin/python $FAstab_dir/data_prepare_script/02_FoldX_BuilModel_mutations_mutiP_sibs.py --repair_pdb_dir ./mutations_structure/wildtype_AF3_preds_repair \
                                                        --out_mutation_dir ./mutations_structure/mutations_FoldX  \
                                                        --mut_csv $mut_csv \
                                                        --foldx5 $foldx5 \
                                                        --process_num 4  # number of parallel processes

######## transfer structures to hdf5 files for further stability prediction
/lustre/home/yhchen/anaconda3/envs/rna_esm/bin/python $FAstab_dir/data_prepare_script/03_read_AF3_cif_FoldX_k50.py --mut_sample_csv $mut_csv \
                             --AF3_pred_dir $AF3_dir \
                             --FoldX_pred_dir ./mutations_structure/mutations_FoldX \
                             --out_hdf5 wildtype_mutations_structures.h5

