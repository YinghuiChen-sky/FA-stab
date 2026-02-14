import os,re,sys,json
import h5py,argparse,logging,tqdm
import esm
import torch
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument("--mutations_sample_csv", type=str, default="./", help="mutations_sample_csv")
parser.add_argument("--AF3_json_dir", type=str, default="./", help="AF3_json_dir")
parser.add_argument("--out_msa_dir", type=str, default="./", help="out_msa_dir")
FLAGS = parser.parse_args()


if os.path.exists(FLAGS.out_msa_dir) == False:
    os.makedirs(FLAGS.out_msa_dir)

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


# wildtype seq
fasta_dict = {}
# get fasta sequences from mutations_sample_csv
df = pd.read_csv(FLAGS.mutations_sample_csv, sep="\t")
for index, row in df.iterrows():
    sample = row['protein']
    sequence = row['wt_seq']
    if sample not in fasta_dict:
        fasta_dict[sample] = sequence

for sample in fasta_dict:
    
    sequence = fasta_dict[sample]
    
    if len(sequence) > 1000:
        print(f"Skipping {sample} due to length > 1000")
        continue


    sample_af3 = re.sub(",", "", sample).lower()
    AF3_json = os.path.join(FLAGS.AF3_json_dir, sample_af3, sample_af3 + "_data.json")
    logging.info(f"Processing AF3 MSA data from {AF3_json}")
    input_a3m = os.path.join(FLAGS.out_msa_dir, sample + "_tmp.a3m")
    no_insert_fasta = os.path.join(FLAGS.out_msa_dir, sample + ".a3m")
    out_a3m = os.path.join(FLAGS.out_msa_dir, sample + "_hhfilter.a3m")

    if not os.path.exists(AF3_json):
        print(f"{AF3_json} not exists, skipping {sample}")
        continue

    with open(AF3_json, "r") as jf:
        data = json.load(jf)
        msa_data = data["sequences"][0]["protein"]["unpairedMsa"]

    # Write to output FASTA
    with open(input_a3m, "w") as out_f:
        out_f.write(f"{msa_data}")

    # Read MSA and remove insertions
    msa_sequences = read_msa(input_a3m, nseq=100000)  # Adjust nseq as needed
    
    # process count matrix
    logging.info("Processing count matrix...")
    msa_tokens_list = []
    
    with open(no_insert_fasta, "w") as out_f:
        for desc, seq in msa_sequences:
            out_f.write(f">{desc}\n{seq}\n")
            msa_tokens = seqs2tokens(seq)
            msa_tokens_list.append(msa_tokens)
    
    msa_tokens_array = np.concatenate(msa_tokens_list, axis=0)  # Shape: (N, L)
    msa_tokens_array = torch.from_numpy(msa_tokens_array).long()
    print("msa_tokens_array", msa_tokens_array.shape)
    count_matrix = torch.zeros(msa_tokens_array.shape[1], 22)
    if msa_tokens_array.shape[1] > 20:
        for i in range(msa_tokens_array.shape[1]):
            count_matrix[i] = torch.bincount(msa_tokens_array[:,i], minlength=22)

    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True))
    # count_matrix = torch.log_softmax(count_matrix, dim=-1)
    
            
    os.system(f"rm {input_a3m}")
    
    logging.info(f"Processed {sample}:  sequences written to {no_insert_fasta}")

    # hhfilter process
    command = f"/lustre/home/yhchen/software/localcolabfold/colabfold-conda/bin/hhfilter -i {no_insert_fasta} -o {out_a3m} -diff 128"
    os.system(command)
    
    logging.info(f"hhfilter completed for {sample}, output written to {out_a3m}")
    
