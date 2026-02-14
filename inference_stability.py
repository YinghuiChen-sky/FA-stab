# -*- coding: utf-8 -*-
import sys,os

import torch
import re,time,random,argparse,h5py,itertools
import esm
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from protein_atoms_fitness_model import Stability_Predict
from utils import getData_fitness_single, infer_test_stability, write_loss_table
from loss import spearman_loss
from metrics import spearman_corr


parser = argparse.ArgumentParser()
parser.add_argument("--coords_seq_hdf5", type=str, default="./", help="path for hdf5 file for coordinates of train/valid samples")
parser.add_argument("--embed_label_h5", type=str, default="./", help="file for fitness label and embedding from ESM2-650M and ESM-1v of train/valid samples")
parser.add_argument("--mutations_csv", type=str, default="./", help="path for Test sample list file")
parser.add_argument("--testset_name", type=str, default="Test", help="name of test set")
parser.add_argument("--k_neighbors", type=int, default=6, help="k_neighbors for e3nn") 
parser.add_argument("--gpu_model", type=int, default=[0,1], nargs='+', help="the IDs of GPU to be used for Fitness/Stab model")
parser.add_argument("--gpu_esm", type=int, default=[1], nargs='+', help="the IDs of GPU to be used for ESM2 model")
parser.add_argument("--cpu", type=int, default=0, help="use cpu to train (1) or not (0)")
parser.add_argument("--Stability_checkpoint", type=str, default="None", help="path of saved parameters of stability model")
parser.add_argument("--output_dir", type=str, default="./", help="Directory for saving models")
FLAGS = parser.parse_args()

if os.path.exists(FLAGS.output_dir) == False:
    os.makedirs(FLAGS.output_dir)

random.seed(42) 

## load data
coords_seq_hdf5 = h5py.File(FLAGS.coords_seq_hdf5, 'r')
embed_label_h5 = h5py.File(FLAGS.embed_label_h5, 'r')

test_mut_list = {}
Test_wild_dict = {}

# load test list from csv file
mut_df = pd.read_csv(FLAGS.mutations_csv, sep="\t")
for index, row in mut_df.iterrows():
    mut_info = row['mut_info']
    wild_name = row['protein']
    mut_name = wild_name + "_" + mut_info
    test_mut_list[mut_name] = wild_name
    if wild_name not in Test_wild_dict:
        Test_wild_dict[wild_name] = [mut_name]
    else:
        Test_wild_dict[wild_name].append(mut_name)

## GPU device
if FLAGS.cpu == 0:
    device_list_model = [ "cuda:"+str(gpu) for gpu in FLAGS.gpu_model ]
    device_list_esm = [ "cuda:"+str(gpu) for gpu in FLAGS.gpu_esm ]
else:
    device_list_model = ["cpu"]
    device_list_esm = ["cpu"]

device1_model = torch.device(device_list_model[0])
device2_model = torch.device(device_list_model[-1])
device_esm =  torch.device(device_list_esm[0])

seq2token_dict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 
                        'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, '-':20, 'X':21} 

amino_acid_list = list('ARNDCQEGHILKMFPSTWYV')
amino_acid_dict = {}
for index, value in enumerate(amino_acid_list):
    amino_acid_dict[index] = value


# Model initialization
Stability_Model = Stability_Predict()


if FLAGS.Stability_checkpoint != "None":
    print("load Stability model from", FLAGS.Stability_checkpoint)
    state = torch.load(FLAGS.Stability_checkpoint, map_location="cpu")
    Stability_Model.load_state_dict(state['net'])

for name, param in Stability_Model.named_parameters():
    param.requires_grad = False

Stability_Model = Stability_Model.to(device=device1_model)
params_model = sum(p.numel() for p in list(Stability_Model.parameters())) / 1e6 # numel()
print('Parameters of FA-stab Model: %.3fM' % (params_model))


################## Test set  ###########################
pred_ddG_dict = {}

test_sample_idx = 0
Stability_Model.eval()
for wild_name_Test in Test_wild_dict.keys():
    test_sample_idx += 1
    mut_name_list = Test_wild_dict[wild_name_Test]
    print("Test wild_name", wild_name_Test, "mut num", len(mut_name_list))

    Test_wild_data = getData_fitness_single(wild_name_Test, coords_seq_hdf5, embed_label_h5, device1_model, device2_model, FLAGS)

    for mut_name_Test in mut_name_list:
        print("Test mut_name", mut_name_Test)
        Test_mut_data = getData_fitness_single(mut_name_Test, coords_seq_hdf5, embed_label_h5, device1_model, device2_model, FLAGS)

        with torch.no_grad(): 
            pred_ddG  = infer_test_stability(Stability_Model, Test_mut_data, Test_wild_data, FLAGS)
            
        pred_ddG_dict[mut_name_Test] = pred_ddG.item()

# write test results
test_output_file = os.path.join(FLAGS.output_dir, FLAGS.testset_name+"_Stability_results.xls")
with open(test_output_file, 'w') as OUT_F:
    OUT_F.write("Mutant\tPred_ddG\n")
    for mut_name in pred_ddG_dict.keys():
        OUT_F.write(mut_name+"\t"+str(pred_ddG_dict[mut_name])+"\n")

coords_seq_hdf5.close()
embed_label_h5.close()