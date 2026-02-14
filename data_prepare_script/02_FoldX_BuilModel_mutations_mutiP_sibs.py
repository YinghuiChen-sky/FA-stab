import sys,os,re,argparse,json
import pandas as pd
import numpy as np
import subprocess
from multiprocessing import Pool
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--repair_pdb_dir", type=str, default=None, help="path of output pdb file")
parser.add_argument("--out_mutation_dir", type=str, default=None, help="path of mutation output directory")
parser.add_argument("--mut_csv", type=str, default=None, help="path of mutation csv file")
parser.add_argument("--process_num", type=int, default=32, help="number of processes to use")
parser.add_argument("--foldx5", type=str, default="/lustre/home/yhchen/software/FoldX/FoldX5/FoldX5_2026/foldx5", help="path of foldx5")
FLAGS = parser.parse_args()

def seqs2tokens(seq): 
    # amino_acid_list = list('ARNDCQEGHILKMFPSTWYV')
    amino_acid_dict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 
                        'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, '-':20, 'X':21} 
    token_list = []
    for i in range(len(seq)):
        res = seq[i]
        if res in amino_acid_dict:
            token_list.append(amino_acid_dict[res])
        else:
            token_list.append(21) # unknown amino acid as 'X'
    token_array = np.array(token_list)
    return token_array



def run_foldx_buildmodel(input_tuple):
    """
    运行单个FoldX buildModel任务
    """
    (protein_name, repair_pdb_dir, mut_list_file, mut_sample_dir, result_pdb) = input_tuple
    try:
        # 构建FoldX命令
        cmd = [
            FLAGS.foldx5, '--command=BuildModel',
            f"--pdb={protein_name}_Repair.pdb",
            f"--pdb-dir={repair_pdb_dir}",
            f"--mutant-file={mut_list_file}",
            f"--output-dir={mut_sample_dir}",
            f"--numberOfRuns=1"
        ]
        
        print(f"Processing: {repair_pdb_dir}/{protein_name}_Repair.pdb with mutations from {mut_list_file}")
        mut_foldx_pdb = os.path.join(mut_sample_dir, protein_name+"_Repair_1.pdb")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(mut_foldx_pdb):
            os.system(f"cp {mut_foldx_pdb} {result_pdb}")
            return {
                'pdb_file': result_pdb,
                'success': True,
                'output': result.stdout
            }
        else:
            return {
                'pdb_file': result_pdb,
                'success': False,
                'error': f"Mutated PDB file not found: {mut_foldx_pdb}"
            }
    except subprocess.CalledProcessError as e:
        return {
            'pdb_file': result_pdb,
            'success': False,
            'error': e.stderr
        }

# 使用示例
if __name__ == "__main__":
    mut_csv = FLAGS.mut_csv
    repair_pdb_dir = FLAGS.repair_pdb_dir
    out_mutation_dir = FLAGS.out_mutation_dir
    process_num = FLAGS.process_num
    
    df_mut = pd.read_csv(mut_csv, sep="\t")

    for i in range(0, len(df_mut), process_num):
        if i + process_num > len(df_mut):
            end_index = len(df_mut)
        else:
            end_index = i + process_num
        batch_idx_range = range(i, end_index)
        
        input_tuples = []
        for j in batch_idx_range:
            mut_info = df_mut.loc[j, "mut_info"]
            protein_name = df_mut.loc[j, "protein"]
            mut_name =protein_name+"_"+mut_info
            wt_seq = df_mut.loc[j, "wt_seq"]
            mut_seq = df_mut.loc[j, "mut_seq"]
            # wt_tokens = seqs2tokens(wt_seq)
            # mut_tokens = seqs2tokens(mut_seq)
            
            # find mutation positions
            # mut_positions = np.where(wt_tokens != mut_tokens)[0]
            mut_positions = mut_info.split(",")
            
            mut_list_info=[]
            for mut in mut_positions:
                mut_list_info.append(mut[0]+"A"+str(int(mut[1:-1]))+mut[-1])

            mut_sample_dir = os.path.join(out_mutation_dir, protein_name, mut_name)
            if not os.path.exists(mut_sample_dir):
                os.system(f"mkdir -p {mut_sample_dir}")
            mut_list_file = os.path.join(mut_sample_dir, "individual_list.txt")
            with open(mut_list_file, 'w') as f:
                for mut in mut_list_info:
                    f.write(mut+";\n")

            mut_foldx_pdb = os.path.join(mut_sample_dir, protein_name+"_Repair_1.pdb")
            result_pdb = os.path.join(out_mutation_dir, protein_name, mut_name+'.pdb')
            input_tuples.append((protein_name, repair_pdb_dir, mut_list_file, mut_sample_dir, result_pdb))
            
        with Pool(processes=process_num) as pool:  # 创建32个进程
            results = pool.map(run_foldx_buildmodel, input_tuples)
            
        # 打印结果摘要
        successful = sum(1 for r in results if r['success'])
        print(f"Completed: {successful}/{len(input_tuples)} successful")
