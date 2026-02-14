import gemmi
import sys,os,re,argparse,h5py,json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--AF3_dir", type=str, help="path of AF3 cif file")
parser.add_argument("--out_pdb", type=str, default=None, help="path of output pdb file")
parser.add_argument("--sample_csv", type=str, default=None, help="path of mutations sample csv file")
parser.add_argument("--out_repair_dir", type=str, default=None, help="path of repair output directory")
parser.add_argument("--foldx5", type=str, default="/lustre/home/yhchen/software/FoldX/FoldX5/FoldX5_2026/foldx5", help="path of foldx5")
FLAGS = parser.parse_args()

def cif_to_pdb_gemmi(cif_file, pdb_file):
    """
    使用gemmi库将CIF文件转换为PDB格式
    
    参数:
    cif_file: 输入的CIF文件路径
    pdb_file: 输出的PDB文件路径
    """


    try:
        # 读取CIF文件
        cif_doc = gemmi.cif.read_file(cif_file)
        
        # 将CIF文档转换为gemmi结构对象
        structure = gemmi.make_structure_from_block(cif_doc.sole_block())
        
        # 写入PDB文件
        structure.write_pdb(pdb_file)
        
        print(f"转换成功: {cif_file} -> {pdb_file}")
        return pdb_file
        
    except Exception as e:
        print(f"转换失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    sample_list = []
    df = pd.read_csv(FLAGS.sample_csv, sep="\t")
    # get unique protein names
    for i in range(len(df)):
        sample = df['protein'][i]
        if sample not in sample_list:
            sample_list.append(sample)
                
    for sample in sample_list:
        AF3_dir = FLAGS.AF3_dir
        out_pdb = FLAGS.out_pdb
        out_repair_dir = FLAGS.out_repair_dir
        sample_af3 = re.sub(",", "", sample).lower()
        af3_cif_path = os.path.join(AF3_dir, sample_af3, sample_af3+"_model.cif")
        pdb_file_1 = os.path.join(AF3_dir, sample_af3, sample_af3 + ".pdb")
        pdb_file_2 = os.path.join(out_pdb, sample+".pdb")
        cif_to_pdb_gemmi(af3_cif_path, pdb_file_1)
        os.system(f"cp {pdb_file_1} {pdb_file_2}")
        
        command_foldx_repair = f"{FLAGS.foldx5} --command=RepairPDB --pdb-dir={out_pdb} --pdb={sample}.pdb --output-dir={out_repair_dir}"
        os.system(command_foldx_repair)
        # repaired_pdb_file = os.path.join(out_pdb, sample+"_Repair.pdb")
        # final_pdb_file = os.path.join(out_pdb, sample+".pdb")