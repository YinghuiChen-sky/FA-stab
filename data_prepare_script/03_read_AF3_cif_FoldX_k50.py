import os,re,sys,json,h5py
import numpy as np
import pandas as pd
from Bio import PDB
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mut_sample_csv", type=str, help="path of mutation csv file")
parser.add_argument("--AF3_pred_dir", type=str, default="", help="path of dir predicted struct pdb files")
parser.add_argument("--FoldX_pred_dir", type=str, default="", help="path of dir predicted struct pdb files")
parser.add_argument("--out_hdf5", type=str, help="path of coords hdf5 file")
FLAGS = parser.parse_args()

seq2token_dict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 
                        'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, '-':20, 'X':21} 

RESIDUE_MAP = {'ALA': 0, 'ARG':1, 'ASN':2, 'ASP':3, 'CYS':4, 'GLN':5, 'GLU':6, 'GLY':7, 'HIS':8, 'ILE':9, 'LEU':10, 
                'LYS':11, 'MET':12, 'PHE':13, 'PRO':14, 'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19, '-':20, 'UNK':21}

ATOM_TYPE_MAP = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

def parse_PDB_to_atoms(cif_path):
    """使用for循环逐行读取原子信息"""
    seq_tokens = []
    atom_coords = []
    atom_types = []
    res_ids = []
    res_types = []
    atom_pLDDT = []
    ca_indices = []
    atom_index = 0
    
    with open(cif_path, 'r') as CIF:
        for line in CIF:
            if line.startswith("ATOM"):
                line = line.strip()
                # parts = re.split(r'\s+', line.strip())
                atom_name = line[12:16].strip()
                atom_type = atom_name[0]  # 使用原子名称的第一个字母作为元素类型
                # atom_index = parts[1]
                if atom_type not in ATOM_TYPE_MAP:
                    continue
                residue_name = line[17:20].strip()
                if residue_name not in RESIDUE_MAP:
                    continue

                res_id = int(line[22:26].strip()) - 1
                plDDT = float(line[60:66].strip())
                atom_pLDDT.append(plDDT)
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_coords.append([x, y, z])
                atom_types.append(ATOM_TYPE_MAP[atom_type])
                res_ids.append(res_id)
                res_types.append(RESIDUE_MAP[residue_name])
                
                if atom_name.strip() == "CA":
                    seq_tokens.append(RESIDUE_MAP[residue_name])
                    ca_indices.append(atom_index)
                atom_index += 1

    return np.array(atom_coords), np.array(atom_types), np.array(atom_pLDDT), np.array(res_ids), np.array(res_types), np.array(seq_tokens), np.array(ca_indices)

def parse_cif_to_atoms(cif_path, chain_id='A'):

    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    model = structure[0]
    chains = list(model.get_chains())
    chain_target = chains[0]

    residues_raw = list(chain_target.get_residues())
    # print("length of residues_raw: ", len(residues_raw))
    # remove non-standard residues
    residue_list = [residue for residue in residues_raw if residue.get_resname() in RESIDUE_MAP.keys()]

    seq_tokens = []
    atom_coords = []
    atom_types = []
    res_ids = []
    res_types = []
    atom_pLDDT = []
    ca_indices = []
    atom_index = 0
    
    for r in range(len(residue_list)):
        residue = residue_list[r]
        residue_name = residue.get_resname()
        if residue_name not in RESIDUE_MAP.keys():
            print("Skipping residue:", residue_name)
            continue
        seq_tokens.append(RESIDUE_MAP[residue_name])
        # print("residue_name", residue_name)
        if residue_name in RESIDUE_MAP.keys():
            atom_list = list(residue.get_atoms())
            for a in range(len(atom_list)):
                atom = atom_list[a]
                atom_type = atom.element
                atom_name =atom.name
                # print("atom_name", atom_name, atom_type)
                if atom_type not in ATOM_TYPE_MAP:
                    continue
                
                coords = atom.get_coord()
                # print(coords)
                pLDDT = round(atom.get_bfactor(),2)
                # print("pLDDT", pLDDT)
                atom_coords.append(coords)
                atom_types.append(ATOM_TYPE_MAP[atom_type])
                #res_ids.append(int(residue.get_id()[1]) - 1)
                res_ids.append(r)
                res_types.append(RESIDUE_MAP[residue_name])
                atom_pLDDT.append(pLDDT)
                
                if atom_name.strip() == "CA":
                    ca_indices.append(atom_index)
                atom_index +=1

    return np.array(atom_coords), np.array(atom_types), np.array(atom_pLDDT), np.array(res_ids), np.array(res_types), np.array(seq_tokens), np.array(ca_indices)

mut_sample_csv_path = FLAGS.mut_sample_csv
mut_sample_list = [] 

df = pd.read_csv(mut_sample_csv_path, sep="\t")
wild_sample_list = list(df['protein'].unique())
for i in range(len(df)):
    mut_name = df['protein'][i] + "_" + df['mut_info'][i]
    if mut_name not in mut_sample_list:
        mut_sample_list.append(mut_name)

FoldX_pred_dir = FLAGS.FoldX_pred_dir
AF3_pred_dir = FLAGS.AF3_pred_dir


OUT_HDF5 = h5py.File(FLAGS.out_hdf5, "w")

for sample in wild_sample_list:
    sample_af3 = re.sub(",","",sample.lower())
    cif_path = f"{AF3_pred_dir}/{sample_af3}/{sample_af3}_model.cif"

    atom_coords, atom_types, atom_pLDDT, residue_ids, residue_types, seq_tokens, ca_indices = parse_cif_to_atoms(cif_path, chain_id='A')

    subgroup = OUT_HDF5.create_group(sample)
    subgroup.create_dataset('target_tokens', data=seq_tokens, dtype=np.int8) #(L,L)
    subgroup.create_dataset('atom_coords', data=atom_coords, dtype=np.float32) #(atoms_num, 3)
    subgroup.create_dataset('atom_types', data=atom_types, dtype=np.int8)  #(atoms_num, )
    subgroup.create_dataset('ca_indices', data=ca_indices, dtype=np.int16)  #(atoms_num, )
    subgroup.create_dataset('atom_pLDDT', data=atom_pLDDT, dtype=np.float32) #(atoms_num, )
    subgroup.create_dataset('residue_mapping', data=residue_ids, dtype=np.int16) #(atoms_num, )
    subgroup.create_dataset('residue_types', data=residue_types, dtype=np.int16) #(atoms_num, )

UNFINISHED = open("unfound_FoldX_pdb.txt", "w")

for sample in mut_sample_list:

    wt_name = sample.rsplit("_",1)[0]
    pdb_path = f"{FoldX_pred_dir}/{wt_name}/{sample}.pdb"
    if os.path.exists(pdb_path) == False:
        print("Not found FoldX pdb file for sample:", sample)
        UNFINISHED.write(f"{sample}\n")
        continue

    atom_coords, atom_types, atom_pLDDT, residue_ids, residue_types, seq_tokens, ca_indices = parse_PDB_to_atoms(pdb_path)
    print("mut_sample", sample, "atom_coords", atom_coords.shape, "atom_types", atom_types.shape, "atom_pLDDT", atom_pLDDT.shape, "residue_ids", residue_ids.shape, "residue_types", residue_types.shape,
            "seq_tokens", seq_tokens.shape, "ca_indices", ca_indices.shape)

    subgroup = OUT_HDF5.create_group(sample)
    subgroup.create_dataset('target_tokens', data=seq_tokens, dtype=np.int8) #(L,L)
    subgroup.create_dataset('atom_coords', data=atom_coords, dtype=np.float32) #(atoms_num, 3)
    subgroup.create_dataset('atom_types', data=atom_types, dtype=np.int8)  #(atoms_num, )
    subgroup.create_dataset('ca_indices', data=ca_indices, dtype=np.int16)  #(atoms_num, )
    subgroup.create_dataset('atom_pLDDT', data=atom_pLDDT, dtype=np.float32) #(atoms_num, )
    subgroup.create_dataset('residue_mapping', data=residue_ids, dtype=np.int16) #(atoms_num, )
    subgroup.create_dataset('residue_types', data=residue_types, dtype=np.int16) #(atoms_num, )
    # subgroup.create_dataset('contact_probs', data=contact_probs, dtype=np.float32) #(L,L)
    # subgroup.create_dataset('pae', data=pae, dtype=np.float32) #(L,L)

UNFINISHED.close()
OUT_HDF5.close()
