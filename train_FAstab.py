# -*- coding: utf-8 -*-
import sys,os
import torch
import re,time,random,argparse,h5py,itertools
import esm
import pandas as pd
import numpy as np
# from scipy.stats import spearmanr

from protein_atoms_fitness_model import Stability_Predict
from utils import getData_fitness_single, thread_Train_valid_stability, thread_Train_valid_stability_reverse, write_loss_table
from loss import spearman_loss
from metrics import spearman_corr


parser = argparse.ArgumentParser()
parser.add_argument("--wt_coords_seq_hdf5", type=str, default="./", help="path for hdf5 file for coordinates of wild-type train/valid samples")
parser.add_argument("--mut_coords_seq_hdf5", type=str, default="./", help="path for hdf5 file for coordinates of mutant train/valid samples")
parser.add_argument("--wt_embed_label_h5", type=str, default="./", help="file for fitness label and embedding from ESM2-650M and ESM-1v of train/valid samples")
parser.add_argument("--mut_embed_label_h5", type=str, default="./", help="file for fitness label and embedding from ESM2-650M and ESM-1v of train/valid samples")

parser.add_argument("--train_csv", type=str, default="./", help="path for k50 sample train list file")
parser.add_argument("--train_k50_list", type=str, default="./", help="path for k50 sample train list file")
parser.add_argument("--valid_k50_list", type=str, default="./", help="path for k50 sample valid list file")

parser.add_argument("--k_neighbors", type=int, default=30, help="k_neighbors for e3nn") 

parser.add_argument("--train_length", type=int, default=256, help="Max lehgth of protein sequence for model training")
parser.add_argument("--valid_length", type=int, default=800, help="Max lehgth of protein sequence for validation")
parser.add_argument("--mini_batch", type=int, default=16, help="mini-batch for spearman loss")
parser.add_argument("--batch", type=int, default=8, help="batch for Gradient Accumulation")
parser.add_argument("--ESM_length_cutoff",  type=int, default=400, help="Max lehgth of protein sequence for ESM2 model")
parser.add_argument("--epoch_start", type=int, default=1, help="number of start training epochs")
parser.add_argument("--epoch_end", type=int, default=30, help="number of end training epochs")
parser.add_argument("--lr_start", type=float, default=0.001, help="learning rate, default=0.001")
parser.add_argument("--lr_end", type=float, default=0.001, help="learning rate, default=0.001")
parser.add_argument("--gpu_model", type=int, default=[0,1], nargs='+', help="the IDs of GPU to be used for Fitness/Stab model")
parser.add_argument("--gpu_esm", type=int, default=[3], nargs='+', help="the IDs of GPU to be used for ESM2 model")
parser.add_argument("--cpu", type=int, default=0, help="use cpu to train (1) or not (0)")
parser.add_argument("--Stability_checkpoint", type=str, default="None", help="path of saved parameters of stability model")
parser.add_argument("--weight_L1_loss", type=float, default=0.1, help="weight for ddG L1 loss")
parser.add_argument("--output_dir", type=str, default="./", help="Directory for saving models")
FLAGS = parser.parse_args()

if os.path.exists(FLAGS.output_dir) == False:
    os.makedirs(FLAGS.output_dir)


random.seed(42)  

## load data
wt_coords_seq_hdf5 = h5py.File(FLAGS.wt_coords_seq_hdf5, 'r')
mut_coords_seq_hdf5 = h5py.File(FLAGS.mut_coords_seq_hdf5, 'r')
wt_embed_label_h5 = h5py.File(FLAGS.wt_embed_label_h5, 'r')
mut_embed_label_h5 = h5py.File(FLAGS.mut_embed_label_h5, 'r')


wild_samples_dict = {}
k50_train_wild_list, k50_valid_wild_list = ([], [])
train_mutations_dict, valid_mutations_dict = ({}, {})

# load k50 train/valid list
with open(FLAGS.train_k50_list, 'r') as LIST:
    for line in LIST:
        sample = re.split("\t",line.strip())[0]
        k50_train_wild_list.append(sample)
with open(FLAGS.valid_k50_list, 'r') as LIST:
    for line in LIST:
        sample = re.split("\t",line.strip())[0]
        k50_valid_wild_list.append(sample)   


# load train/valid list
ddG_train = pd.read_csv(FLAGS.train_csv, sep=",")
rows = len(ddG_train)
for i in range(rows):
    wild_name = ddG_train.loc[i,"protein_name"]
    mut_info = ddG_train.loc[i,"mut_info"]
    mut_name = f"{wild_name}_{mut_info}"
    if wild_name in k50_train_wild_list:
        train_mutations_dict[mut_name] = wild_name
    elif wild_name in k50_valid_wild_list:
        valid_mutations_dict[mut_name] = wild_name
    # else:
    #     print("Error: not in train or valid list", mut_name)

    if wild_name not in wild_samples_dict:
        wild_samples_dict[wild_name] = [mut_name]
    else:
        wild_samples_dict[wild_name].append(mut_name)
        
print("train mut samples", len(train_mutations_dict), "valid mut samples", len(valid_mutations_dict))
print("train wild samples", len(k50_train_wild_list), "valid wild samples", len(k50_valid_wild_list) )


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

def create_loss_dict(sample_name=False):
    dict = {"epoch":[], "batch":[], "sample_name":[],
    "spearman_loss":[], "spearman_corr":[], "L1_loss":[], "ddG_pred":[],  "ddG_true":[], "L1_loss_rev":[]
    }
    if sample_name == False:
        del dict['batch']
        del dict['sample_name']
    else:
        del dict['epoch']
    return dict

def create_L1_loss_dict():
    mut_sample_dict = {"sample_name":[], "L1_loss":[], "ddG_pred":[],  "ddG_true":[], "L1_loss_rev":[]}
    return mut_sample_dict

amino_acid_list = list('ARNDCQEGHILKMFPSTWYV')
amino_acid_dict = {}
for index, value in enumerate(amino_acid_list):
    amino_acid_dict[index] = value

double_mut_list = list(itertools.product(amino_acid_list, amino_acid_list, repeat = 1))
double_mut_dict = {}
double_mut_dict_inverse = {}
for index, value in enumerate(double_mut_list):
    double_mut_dict[index] = ''.join(value)
    double_mut_dict_inverse[''.join(value)] = index

######################### Train and Validation Process  ###########################

# Model initialization
Stability_Model = Stability_Predict()

if FLAGS.Stability_checkpoint != "None":
    print("load Stability model from", FLAGS.Stability_checkpoint)
    state = torch.load(FLAGS.Stability_checkpoint, map_location="cpu")
    Stability_Model.load_state_dict(state['net'])

# frozen parameters except the ddG and dTm prediction layers
for name, param in Stability_Model.named_parameters():
    if "finetune_ddG_coef" in name or "dTm" in name:
        param.requires_grad = False
        print("Frozen param", name, param.shape)
    else:
        param.requires_grad = True
        print("Trainable param", name, param.shape)
    

Stability_Model = Stability_Model.to(device=device1_model)
params_model = sum(p.numel() for p in list(Stability_Model.parameters())) / 1e6 # numel()
print('Parameters of Stability_Model Model: %.3fM' % (params_model))

optimizer = torch.optim.Adam(Stability_Model.parameters(), lr=FLAGS.lr_start)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5, verbose=True)

Loss_train_all_epochs, Loss_valid_all_epochs = (create_loss_dict(sample_name=False),  create_loss_dict(sample_name=False))
stop_step = 0
best_loss = float("inf")
early_stop = 10

## each epoch
for epoch in range(FLAGS.epoch_start, FLAGS.epoch_end + 1):

    Loss_train_epoch, Loss_valid_epoch = (create_loss_dict(sample_name=True),  create_loss_dict(sample_name=True))
    Loss_train_samples,  Loss_valid_samples = (create_L1_loss_dict(), create_L1_loss_dict())

    # shuffle train data set
    train_samples_list = list(train_mutations_dict.keys())
    random.shuffle(train_samples_list)

    ## set Generator into training mode
    Stability_Model.train()

    ## training
    epoch_loss_train = 0
    train_batch_idx = 0
    sample_index = 0
    for idx_train in range(0, len(train_samples_list)-1, FLAGS.mini_batch):
        train_batch_idx += 1
        print("sample_index", sample_index)

        batch_list = train_samples_list[idx_train : idx_train+FLAGS.mini_batch]

        print( "batch_sample_num", len(batch_list))
        
        pred_ddG_list, pred_dTm_list, label_ddG_list, label_dTm_list = ([], [], [], [])
        ddG_L1_Loss_list = []
        ddG_L1_Loss_rev_list = []

        for mut_name_train in batch_list:
            sample_index += 1

            wild_name_train = train_mutations_dict[mut_name_train]
            print("mut_name_train", mut_name_train, "wild_name_train", wild_name_train)

            train_mut_data = getData_fitness_single(mut_name_train, mut_coords_seq_hdf5, mut_embed_label_h5, device1_model, device2_model, FLAGS)
            train_wild_data = getData_fitness_single(wild_name_train, wt_coords_seq_hdf5, wt_embed_label_h5, device1_model, device2_model, FLAGS)
            print("train_wild_data", train_wild_data["target_tokens"].shape, "train_mut_data", train_mut_data["target_tokens"].shape)
            
            # Model infer and train
            ddG_L1_loss, dTm_L1_loss, pred_ddG, pred_dTm, label_ddG, label_dTm  = thread_Train_valid_stability(Stability_Model, train_mut_data, train_wild_data, FLAGS) 
            ddG_L1_loss_rev, dTm_L1_loss_rev, pred_ddG_rev, pred_dTm_rev, label_ddG_rev, label_dTm_rev = thread_Train_valid_stability_reverse(Stability_Model, train_mut_data, train_wild_data, FLAGS) 
            
            # print("pred_ddG", pred_ddG.shape, "label_ddG", label_ddG.shape)
            pred_ddG_list.append(pred_ddG)
            label_ddG_list.append(label_ddG)
            ddG_L1_Loss_list.append(ddG_L1_loss)
            ddG_L1_Loss_rev_list.append(ddG_L1_loss_rev)

            Loss_train_samples["sample_name"].append(mut_name_train)
            Loss_train_samples["ddG_pred"].append(pred_ddG.item())
            Loss_train_samples["ddG_true"].append(label_ddG.item())
            Loss_train_samples["L1_loss"].append(ddG_L1_loss.item())
            Loss_train_samples["L1_loss_rev"].append(ddG_L1_loss_rev.item())

            ## Loss SUM
            Loss_sum = FLAGS.weight_L1_loss * ((ddG_L1_loss + ddG_L1_loss_rev) / 2)
            ## loss backward
            epoch_loss_train += Loss_sum.item()
            Loss_sum = Loss_sum / (FLAGS.batch * FLAGS.mini_batch)  # Loss Normlization for Gradient Accumulation
            Loss_sum.backward()

        pred_ddG_tensor = torch.cat(pred_ddG_list, dim=0).unsqueeze(0)
        label_ddG_tensor = torch.stack(label_ddG_list).unsqueeze(0)
        ddG_L1_Loss_tensor = torch.stack(ddG_L1_Loss_list)
        ddG_L1_Loss_tensor_rev = torch.stack(ddG_L1_Loss_rev_list)
        #print("pred_ddG_tensor", pred_ddG_tensor.shape, "label_ddG_tensor", label_ddG_tensor.shape, "ddG_L1_Loss_tensor", ddG_L1_Loss_tensor.shape)

        spearman_loss_ddG = spearman_loss(pred_ddG_tensor, label_ddG_tensor, 1e-2, 'kl')
        spearman_corr_ddG = spearman_corr(pred_ddG_tensor.squeeze(0), label_ddG_tensor.squeeze(0))
        ddG_L1_Loss_mean = torch.nanmean(ddG_L1_Loss_tensor)
        ddG_L1_Loss_mean_rev = torch.nanmean(ddG_L1_Loss_tensor_rev)

        Loss_train_epoch["batch"].append(train_batch_idx)
        Loss_train_epoch["sample_name"].append(wild_name)
        Loss_train_epoch["ddG_pred"].append(pred_ddG_tensor.mean().item())
        Loss_train_epoch["ddG_true"].append(label_ddG_tensor.mean().item())
        Loss_train_epoch["spearman_loss"].append(spearman_loss_ddG.item())
        Loss_train_epoch["spearman_corr"].append(spearman_corr_ddG.item())
        Loss_train_epoch["L1_loss"].append(ddG_L1_Loss_mean.item())
        Loss_train_epoch["L1_loss_rev"].append(ddG_L1_Loss_mean_rev.item())

        print("ddG_pred", round(pred_ddG.item(),3), "ddG_true", round(train_mut_data["ddG"].item(),3) if train_mut_data["ddG"] is not None else "NA")
        print( "spearman_loss_ddG", spearman_loss_ddG.item(), "spearman_corr", spearman_corr_ddG.item(), "ddG_L1_Loss_mean",ddG_L1_Loss_mean.item())
        

        if train_batch_idx % FLAGS.batch == 0 :
            # torch.nn.utils.clip_grad_norm_(Stability_Model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
            optimizer.step() 
            optimizer.zero_grad()
        
        
        if sample_index % 10112 == 0 or sample_index % len(train_samples_list) == 0:
            ## save model
            state = {'net':Stability_Model.state_dict(), 'optimizer':optimizer.state_dict()}
            torch.save(state, os.path.join(FLAGS.output_dir,"Model_epoch{}_{}.pth".format(epoch, sample_index)))

            ## loss detail
            Loss_train_all_epochs["epoch"].append(epoch)
            for key, loss_list in Loss_train_epoch.items():
                if key not in ["batch","sample_name"]:
                    Loss_train_all_epochs[key].append(np.nanmean(loss_list))
            write_loss_table(data_dict=Loss_train_all_epochs, output_dir=FLAGS.output_dir, file_name="Loss_train_mean_epoch{}.xls".format(epoch))
            write_loss_table(data_dict=Loss_train_epoch, output_dir=FLAGS.output_dir, file_name="Loss_train_spearman_epoch{}.xls".format(epoch))
            write_loss_table(data_dict=Loss_train_samples, output_dir=FLAGS.output_dir, file_name="Loss_train_pred_epoch{}.xls".format(epoch))

    epoch_loss_train = epoch_loss_train / len(train_samples_list)
    print("epoch_loss_train", epoch_loss_train)
    
    ################## validation  ###########################
    valid_sample_index = 0
    valid_batch_index = 0
    epoch_loss_validation = 0
    Stability_Model.eval()
    valid_samples_list = list(valid_mutations_dict.keys())

    print("valid samples", len(valid_samples_list))
    for valid_idx in range(0, len(valid_samples_list)-1, FLAGS.mini_batch):
        valid_batch_index += 1

        valid_batch_list = valid_samples_list[valid_idx:valid_idx+FLAGS.mini_batch]
        
        pred_ddG_list, pred_dTm_list, label_ddG_list, label_dTm_list = ([], [], [], [])
        ddG_L1_Loss_list = []
        ddG_L1_Loss_rev_list = []

        for mut_name_valid in valid_batch_list:
            valid_sample_index += 1
            wild_name_valid = valid_mutations_dict[mut_name_valid]
            
            valid_wild_data = getData_fitness_single(wild_name_valid, wt_coords_seq_hdf5, wt_embed_label_h5, device1_model, device2_model, FLAGS)
            valid_mut_data = getData_fitness_single(mut_name_valid, mut_coords_seq_hdf5, mut_embed_label_h5, device1_model, device2_model, FLAGS)

            with torch.no_grad(): 
                ddG_L1_loss, dTm_L1_loss, pred_ddG, pred_dTm, label_ddG, label_dTm  = thread_Train_valid_stability(Stability_Model, valid_mut_data, valid_wild_data, FLAGS) 
                ddG_L1_loss_rev, dTm_L1_loss_rev, pred_ddG_rev, pred_dTm_rev, label_ddG_rev, label_dTm_rev = thread_Train_valid_stability_reverse(Stability_Model, valid_mut_data, valid_wild_data, FLAGS) 

            pred_ddG_list.append(pred_ddG)
            label_ddG_list.append(label_ddG)
            ddG_L1_Loss_list.append(ddG_L1_loss)
            ddG_L1_Loss_rev_list.append(ddG_L1_loss_rev)
            Loss_valid_samples["sample_name"].append(mut_name_valid)
            Loss_valid_samples["ddG_pred"].append(pred_ddG.item())
            Loss_valid_samples["ddG_true"].append(label_ddG.item())
            Loss_valid_samples["L1_loss"].append(ddG_L1_loss.item())
            Loss_valid_samples["L1_loss_rev"].append(ddG_L1_loss_rev.item())
            
            Loss_sum = FLAGS.weight_L1_loss * ((ddG_L1_loss + ddG_L1_loss_rev) / 2)
            epoch_loss_validation += Loss_sum.item()
            
        pred_ddG_tensor = torch.cat(pred_ddG_list, dim=0).unsqueeze(0)
        label_ddG_tensor = torch.stack(label_ddG_list).unsqueeze(0)
        ddG_L1_Loss_tensor = torch.stack(ddG_L1_Loss_list)
        ddG_L1_Loss_tensor_rev = torch.stack(ddG_L1_Loss_rev_list)
        #print("pred_ddG_tensor", pred_ddG_tensor.shape, "label_ddG_tensor", label_ddG_tensor.shape, "ddG_L1_Loss_tensor", ddG_L1_Loss_tensor.shape, "ddG_L1_Loss_tensor_rev", ddG_L1_Loss_tensor_rev.shape)

        spearman_loss_ddG = spearman_loss(pred_ddG_tensor, label_ddG_tensor, 1e-2, 'kl')
        spearman_corr_ddG = spearman_corr(pred_ddG_tensor.squeeze(0), label_ddG_tensor.squeeze(0))
        ddG_L1_Loss_mean = torch.nanmean(ddG_L1_Loss_tensor)
        ddG_L1_Loss_mean_rev = torch.nanmean(ddG_L1_Loss_tensor_rev)
        Loss_valid_epoch["batch"].append(valid_batch_index)
        Loss_valid_epoch["sample_name"].append(wild_name_valid)
        Loss_valid_epoch["ddG_pred"].append(pred_ddG_tensor.mean().item())
        Loss_valid_epoch["ddG_true"].append(label_ddG_tensor.mean().item())
        Loss_valid_epoch["L1_loss"].append(ddG_L1_Loss_mean.item())
        Loss_valid_epoch["L1_loss_rev"].append(ddG_L1_Loss_mean_rev.item())
        Loss_valid_epoch["spearman_loss"].append(spearman_loss_ddG.item())
        Loss_valid_epoch["spearman_corr"].append(spearman_corr_ddG.item())

    epoch_loss_validation = epoch_loss_validation / valid_sample_index
    print("epoch_loss_valid", epoch_loss_validation)
    scheduler.step(epoch_loss_validation)


    Loss_valid_all_epochs["epoch"].append(epoch)
    for key, loss_list in Loss_valid_epoch.items():
        if key not in ["batch","sample_name"]:
            Loss_valid_all_epochs[key].append(np.nanmean(loss_list))
    write_loss_table(data_dict=Loss_valid_all_epochs, output_dir=FLAGS.output_dir, file_name="Loss_valid_mean_epoch{}.xls".format(epoch))
    write_loss_table(data_dict=Loss_valid_epoch, output_dir=FLAGS.output_dir, file_name="Loss_valid_spearman_epoch{}.xls".format(epoch))
    write_loss_table(data_dict=Loss_valid_samples, output_dir=FLAGS.output_dir, file_name="Loss_valid_pred_epoch{}.xls".format(epoch))


    if epoch_loss_validation < best_loss:
        stop_step = 0
        best_loss = epoch_loss_validation
    else:
        stop_step += 1
        print("stop_step", stop_step)
        if stop_step >= early_stop:
            print("early_stop", "stop_step", stop_step, "early_stop", early_stop)
            break


wt_coords_seq_hdf5.close()
mut_coords_seq_hdf5.close()
wt_embed_label_h5.close()
mut_embed_label_h5.close()
