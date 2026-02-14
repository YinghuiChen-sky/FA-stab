import os,re,sys,json
import h5py,argparse,logging
import esm
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="./", help="device")
parser.add_argument("--filtered_msa_dir", type=str, default="./", help="output msa directory")
parser.add_argument("--esm_model_dir", type=str, default="./", help="esm_model_dir")
parser.add_argument("--out_hdf5_path", type=str, default="./", help="hdf5 path for wild type")
parser.add_argument("--ddG_train_data_csv", type=str, default="./", help="ddG_train_data")
FLAGS = parser.parse_args()

device = torch.device(FLAGS.device)
esm_model_dir = FLAGS.esm_model_dir
OUT_HDF5 = h5py.File(FLAGS.out_hdf5_path, "w")
# WT_HDF5 = h5py.File(FLAGS.wt_hdf5_path, "w")
# MUT_HDF5 = h5py.File(FLAGS.mt_hdf5_path, "w")

ddG_train = pd.read_csv(FLAGS.ddG_train_data_csv, sep="\t")

# seq2token_dict = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 
#                 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19,  'U':20}

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
    token_array = np.array(token_list)[None, :]
    return token_array

# This is an efficient way to delete lowercase characters and insertion characters from a string
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


def get_af3_mutation_train_json(dataframe, wild_dict, mt_dict):
    rows = len(dataframe)
    for i in range(rows):
        mut_info = dataframe.loc[i,"mut_info"]
        wild_name = dataframe.loc[i,"protein"]
        mut_name = f"{wild_name}_{mut_info}"
        
        wt_seq = dataframe.loc[i,"wt_seq"]
        mut_seq = dataframe.loc[i,"mut_seq"]
        # ddG_label = dataframe.loc[i,"ddG"]
        
        # mut_seq_tokens = seqs2tokens(mut_seq, seq2token_dict)
        # wt_seq_tokens = seqs2tokens(wt_seq, seq2token_dict)

        # diff = mut_seq_tokens - wild_seq_tokens
        # indices = np.where(diff != 0)[0]

        if wild_name not in wild_dict:
            wild_dict[wild_name] = wt_seq
        if mut_name not in mt_dict:
            mt_dict[mut_name] = {"mut_seq":mut_seq,  "wild_name":wild_name,} #"label":ddG_label,
        
    return wild_dict, mt_dict


def gen_esm1v_logits(model, batch_tokens, alphabet, seq_data):

    with torch.no_grad():
        
        seq = seq_data[0][1]
        
        token_probs = torch.log_softmax(model(batch_tokens)['logits'], dim = -1)
        logits_33 = token_probs[0, 1:-1, :].detach().cpu().clone()
        
        # logits 33 dim -> 20 dim
        amino_acid_list = list('ARNDCQEGHILKMFPSTWYV')
        esm1v_amino_acid_dict = {}
        for i in amino_acid_list:
            esm1v_amino_acid_dict[i] = alphabet.get_idx(i)
        
        logits_20_single = torch.zeros((logits_33.shape[0], 20))
        for wt_pos, wt_amino_acid in enumerate(seq):
            for mut_pos, mut_amino_acid in enumerate(amino_acid_list):
                logits_20_single[wt_pos, mut_pos] = logits_33[wt_pos, esm1v_amino_acid_dict[mut_amino_acid]] - logits_33[wt_pos, esm1v_amino_acid_dict[wt_amino_acid]]
        
        logits_20_double = (logits_20_single[:, None, :, None] + logits_20_single[None, :, None, :]).reshape(len(seq), len(seq), 20 * 20)
        return (logits_20_single.unsqueeze(0), logits_20_double.unsqueeze(0))

wild_dict, mt_dict = ({}, {})
wild_sample_msa_dict = {}
wild_dict, mt_dict = get_af3_mutation_train_json(ddG_train, wild_dict, mt_dict)

# wild_dict, mt_dict = get_af3_mutation_test_json(ddG_S461_test, "ddG", wild_dict, mt_dict)
# wild_dict, mt_dict = get_af3_mutation_test_json(ddG_S669_test, "ddG", wild_dict, mt_dict)
# wild_dict, mt_dict = get_af3_mutation_test_json(dTm_S557_test, "dTm", wild_dict, mt_dict)

print("wild_dict", len(wild_dict), "mt_dict", len(mt_dict))

esm_650M_path = f"{esm_model_dir}/esm2_t33_650M_UR50D.pt"
esm2_650M, alphabet_esm2 = esm.pretrained.load_model_and_alphabet(esm_650M_path)
batch_converter_esm2 = alphabet_esm2.get_batch_converter()
esm2_650M.eval().to(device)

# load 5 ESM-1v models
esm1v_1 = f"{esm_model_dir}/esm1v_t33_650M_UR90S_1.pt"
esm1v_2 = f"{esm_model_dir}/esm1v_t33_650M_UR90S_2.pt"
esm1v_3 = f"{esm_model_dir}/esm1v_t33_650M_UR90S_3.pt"
esm1v_4 = f"{esm_model_dir}/esm1v_t33_650M_UR90S_4.pt"
esm1v_5 = f"{esm_model_dir}/esm1v_t33_650M_UR90S_5.pt"

esm1v_1, _ = esm.pretrained.load_model_and_alphabet(esm1v_1)
esm1v_2, _ = esm.pretrained.load_model_and_alphabet(esm1v_2)
esm1v_3, _ = esm.pretrained.load_model_and_alphabet(esm1v_3)
esm1v_4, _ = esm.pretrained.load_model_and_alphabet(esm1v_4)
esm1v_5, esm1v_alphabet= esm.pretrained.load_model_and_alphabet(esm1v_5)
esm1v_batch_converter = esm1v_alphabet.get_batch_converter()
# esm1v_5, esm1v_alphabet = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_5")
esm1v_1.eval().to(device)
esm1v_2.eval().to(device)
esm1v_3.eval().to(device)
esm1v_4.eval().to(device)
esm1v_5.eval().to(device)

esm_1b_ckp = f"{esm_model_dir}/esm_msa1b_t12_100M_UR50S.pt"
esm_1b, alphabet = esm.pretrained.load_model_and_alphabet(esm_1b_ckp)
batch_converter_esm1b = alphabet.get_batch_converter()
esm_1b.eval()
esm_1b = esm_1b.to(device)


for sample in wild_dict:
    
    print("sample", sample)
    wild_seq = wild_dict[sample]
    print("wild_seq", wild_seq)
    wild_seq_tokens = seqs2tokens(wild_seq)
    seq_data = [(sample, wild_seq)]
    # print(seq_data)
    _, strs, target_tokens = batch_converter_esm2(seq_data)
    target_tokens = target_tokens.to(device)
    
    with torch.no_grad():
        result_esm2_650m = esm2_650M(target_tokens, repr_layers = [33], return_contacts = False)
        
    f1d_esm2_650M = result_esm2_650m['representations'][33][0, 1:-1, :].to(device) #(L,1280)

    esm1v_single_1, esm1v_double_1 = gen_esm1v_logits(esm1v_1, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_2, esm1v_double_2 = gen_esm1v_logits(esm1v_2, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_3, esm1v_double_3 = gen_esm1v_logits(esm1v_3, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_4, esm1v_double_4 = gen_esm1v_logits(esm1v_4, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_5, esm1v_double_5 = gen_esm1v_logits(esm1v_5, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_logits = torch.cat([esm1v_single_1, esm1v_single_2, esm1v_single_3, esm1v_single_4, esm1v_single_5 ],dim=0).to(device) #(5,L,20)
    esm1v_double_logits = torch.cat([esm1v_double_1, esm1v_double_2, esm1v_double_3, esm1v_double_4, esm1v_double_5 ],dim=0).to(device) #(5,L,L,400)


    f1d_esm2_650M = f1d_esm2_650M.cpu().numpy() #(L,1280)
    esm1v_single_logits = esm1v_single_logits.cpu().numpy() #(5,L,20)
    print("f1d_esm2_650M", f1d_esm2_650M.shape,  "esm1v_single_logits", esm1v_single_logits.shape)

    subgroup = OUT_HDF5.create_group(sample)
    subgroup.create_dataset('seq_tokens', data=wild_seq_tokens, dtype=np.float32)
    subgroup.create_dataset('esm2_650M', data=f1d_esm2_650M, dtype=np.float32)
    subgroup.create_dataset('esm1v_single_logits', data=esm1v_single_logits, dtype=np.float32)

    out_a3m = os.path.join(FLAGS.filtered_msa_dir, sample + "_hhfilter.a3m")
    
    # 2. 准备输入数据
    logging.info(f"Preparing MSA data from {out_a3m}")
    sequences = []
    for record in SeqIO.parse(out_a3m, "fasta"):
        sequences.append((record.id, str(record.seq)))
    
    # 确保序列数量不超过1024
    if len(sequences) > 129:
        sequences = sequences[:129]
        logging.warning(f"Truncated MSA to first 129 sequences")
    
    # 3. 转换为模型输入
    _, _, batch_tokens = batch_converter_esm1b(sequences)
    batch_tokens = batch_tokens.to(device)
    
    # 4. 生成嵌入
    logging.info("Generating embeddings...")
    with torch.no_grad():
        results = esm_1b(batch_tokens, repr_layers=[12], need_head_weights=True, return_contacts=True)
    
    msa_embeddings = results["representations"][12].cpu().numpy() # [1, MSA, L+1, 768]
    target_embedding = msa_embeddings[0, 0, 1:, :] # [L, 768]

    # contact = results["contacts"].cpu().numpy() # [1, L, L]
    # row_attentions = results["row_attentions"][:, :, :, 1:, 1:].cpu().numpy()  # [1, 12, 12, L, L]
    
    subgroup.create_dataset("msa_transformer", data=target_embedding, dtype=np.float32)
    # subgroup.create_dataset("msa_count_matrix", data=count_matrix.cpu().numpy(), dtype=np.float32)


for sample in mt_dict:
    
    print("mut_sample", sample)
    
    mut_seq = mt_dict[sample]["mut_seq"]
    wild_sample_name = mt_dict[sample]["wild_name"]
    wild_seq = wild_dict[wild_sample_name]
    mut_seq_tokens = seqs2tokens(mut_seq)
    # ddG_label = mt_dict[sample]["label"]

    seq_data = [(sample, mut_seq)]
    labels, strs, target_tokens = batch_converter_esm2(seq_data)
    target_tokens = target_tokens.to(device)
    
    with torch.no_grad():
        result_esm2_650m = esm2_650M(target_tokens, repr_layers = [33], return_contacts = False)
        
    f1d_esm2_650M = result_esm2_650m['representations'][33][0, 1:-1, :].to(device) #(L,1280)

    esm1v_single_1, esm1v_double_1 = gen_esm1v_logits(esm1v_1, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_2, esm1v_double_2 = gen_esm1v_logits(esm1v_2, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_3, esm1v_double_3 = gen_esm1v_logits(esm1v_3, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_4, esm1v_double_4 = gen_esm1v_logits(esm1v_4, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_5, esm1v_double_5 = gen_esm1v_logits(esm1v_5, target_tokens, esm1v_alphabet, seq_data)
    esm1v_single_logits = torch.cat([esm1v_single_1, esm1v_single_2, esm1v_single_3, esm1v_single_4, esm1v_single_5 ],dim=0).to(device) #(5,L,20)
    esm1v_double_logits = torch.cat([esm1v_double_1, esm1v_double_2, esm1v_double_3, esm1v_double_4, esm1v_double_5 ],dim=0).to(device) #(5,L,L,400)


    f1d_esm2_650M = f1d_esm2_650M.cpu().numpy() #(L,1280)
    esm1v_single_logits = esm1v_single_logits.cpu().numpy() #(5,L,20)
    # print("f1d_esm2_650M", f1d_esm2_650M.shape,  "esm1v_single_logits", esm1v_single_logits.shape)

    subgroup = OUT_HDF5.create_group(sample)
    subgroup.create_dataset('seq_tokens', data=mut_seq_tokens, dtype=np.float32)
    subgroup.create_dataset('esm2_650M', data=f1d_esm2_650M, dtype=np.float32)
    subgroup.create_dataset('esm1v_single_logits', data=esm1v_single_logits, dtype=np.float32)
    # subgroup.create_dataset('ddG_label', data=ddG_label, dtype=np.float32)

    out_a3m = os.path.join(FLAGS.filtered_msa_dir, wild_sample_name + "_hhfilter.a3m")
    
    # 2. 准备输入数据
    logging.info(f"Preparing MSA data from {out_a3m}")
    sequences = []
    for record in SeqIO.parse(out_a3m, "fasta"):
        if record.id == "query":
            sequences.append((record.id, mut_seq))
            print("use mutant sequence for query")
        sequences.append((record.id, str(record.seq)))
    
    # 确保序列数量不超过129
    if len(sequences) > 129:
        sequences = sequences[:129]
        logging.warning(f"Truncated MSA to first 129 sequences")
    
    # 3. 转换为模型输入
    _, _, batch_tokens = batch_converter_esm1b(sequences)
    batch_tokens = batch_tokens.to(device)
    
    # 4. 生成嵌入
    logging.info("Generating embeddings...")
    with torch.no_grad():
        results = esm_1b(batch_tokens, repr_layers=[12], need_head_weights=True, return_contacts=True)
    
    msa_embeddings = results["representations"][12].cpu().numpy() # [1, MSA, L+1, 768]
    target_embedding = msa_embeddings[0, 0, 1:, :] # [L, 768]

    # contact = results["contacts"].cpu().numpy() # [1, L, L]
    # row_attentions = results["row_attentions"][:, :, :, 1:, 1:].cpu().numpy()  # [1, 12, 12, L, L]
    
    subgroup.create_dataset("msa_transformer", data=target_embedding, dtype=np.float32)
    # subgroup.create_dataset("msa_count_matrix", data=count_matrix.cpu().numpy(), dtype=np.float32)


OUT_HDF5.close()