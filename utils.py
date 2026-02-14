import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os,sys,time,random
from metrics import spearman_corr
from collections import defaultdict

# 图构建函数 (K最近邻)
def build_knn_graph(atom_coords, k=20):
    dist = torch.cdist(atom_coords, atom_coords)
    _, indices = torch.topk(dist, k=k, dim=1, largest=False)
    row = torch.arange(atom_coords.shape[0], device=atom_coords.device).repeat_interleave(k)
    col = indices.flatten()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def build_distance_limited_knn_graph(atom_coords, k=6, max_distance=8.0):
    """
    高效构建带距离限制的K最近邻图
    
    参数:
        pos: 原子坐标 [num_nodes, 3]
        k: 最大邻居数
        max_distance: 最大连接距离 (埃)
    
    返回:
        edge_index: 边索引 [2, num_edges]
    """
    # 计算所有原子对之间的距离
    dist = torch.cdist(atom_coords, atom_coords)
    
    # 创建距离掩码 (小于max_distance且排除自身)
    mask = (dist < max_distance) & (dist > 0.1)
    
    # 初始化边列表
    rows = []
    cols = []
    
    # 对于每个原子，找到满足条件的邻居
    for i in range(atom_coords.shape[0]):
        # 获取当前原子的有效邻居索引
        neighbors = torch.where(mask[i])[0]
        
        if len(neighbors) > 0:
            # 获取这些邻居的距离
            neighbor_dists = dist[i, neighbors]
            
            # 如果邻居数量超过k，只保留最近的k个
            if len(neighbors) > k:
                # 获取前k个最小距离的索引
                _, topk_indices = torch.topk(neighbor_dists, k=k, largest=False)
                selected = neighbors[topk_indices]
            else:
                selected = neighbors
            
            # 添加边
            rows.extend([i] * len(selected))
            cols.extend(selected.tolist())
    
    # 转换为边索引张量
    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return edge_index

def tokens2seq(token_array):
    # X: rare amino acid or unknown amino acid;  "-": gap; 
    # amino_acid_dict = {0:'A', 1:'C', 2:'D', 3:'E', 4:'F', 5:'G', 6:'H', 7:'I', 8:'K', 9:'L', 10:'M', 11:'N', 12:'P', 
    #                     13:'Q', 14:'R', 15:'S', 16:'T', 17:'V', 18:'W', 19:'Y', 20:'X', 21:'O', 22:'U', 23:'B', 24:'Z', 25:'-', 26:'.', 27:'<mask>', 28: '<pad>',} 
    amino_acid_dict = {0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I', 10: 'L', 11: 'K', 
                        12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V', 20: '-', 21: 'X'}
    msa_sequence_list = []
    row,col = token_array.shape
    for i in range(row):
        token_list = []
        for k in range(col):
            token = amino_acid_dict[token_array[i,k]]
            token_list.append(token)
        sequence =  "".join(token_list)
        msa_sequence_list.append((str(i), sequence))
    return(msa_sequence_list)

def ESM2_embed(seq_tokens, batch_converter_esm2, model_CPU, model_GPU, length_cutoff, gpu):
    # pred_seq = (batch, L) differentialbel one-hot
    batch, L = seq_tokens.shape
    L = L - 2
    with torch.no_grad():
        if L > length_cutoff:
            print("L",L,"length_cutoff", length_cutoff, "CPU")
            results = model_CPU(seq_tokens.to(device="cpu"), repr_layers=range(37), need_head_weights=False, return_contacts=True)
        else:
            print("L",L,"length_cutoff", length_cutoff, "GPU")
            results = model_GPU(seq_tokens.to(device=gpu), repr_layers=range(37), need_head_weights=False, return_contacts=True)
        
    token_embeds = torch.stack([v for _, v in sorted(results["representations"].items())], dim=2)
    token_embeds = token_embeds[:, 1:-1] # (batch, L, dim=2560)
    token_embeds = token_embeds.to(device=gpu, dtype=torch.float32) # (batch, L, dim=2560)

    ###### attention map and contact map ######
    attentions = results["attentions"] #(batch, layers=36, heads=40, L+2, L+2)
    attentions = attentions[:, -1, :, 1:-1, 1:-1] #(batch, 40, L, L)
    contacts = results["contacts"].unsqueeze(1) # (batch, 1, L, L)
    return (token_embeds, attentions)


def getData_fitness_single(sample, coords_hdf5, fitness_h5, device1, device2, FLAGS):
    s0 = time.time()
    target_tokens = coords_hdf5[sample]['target_tokens'][:][None,:] # (1,L)
    seq_str = tokens2seq(target_tokens)
    target_tokens = torch.from_numpy(target_tokens).to(dtype=torch.long)
    L = target_tokens.shape[-1]


    atom_coords = torch.from_numpy(coords_hdf5[sample]['atom_coords'][:]).to(dtype=torch.float32) # (atoms_num, 3)
    atom_types = torch.from_numpy(coords_hdf5[sample]['atom_types'][:]).to(dtype=torch.long) #(atoms_num, )
    atom_pLDDT = torch.from_numpy(coords_hdf5[sample]['atom_pLDDT'][:]).to(dtype=torch.float32) #(atoms_num, )
    residue_types = torch.from_numpy(coords_hdf5[sample]['residue_types'][:]).to(dtype=torch.long) #(atoms_num, )
    residue_mapping = torch.from_numpy(coords_hdf5[sample]['residue_mapping'][:]).to(dtype=torch.long) #(atoms_num, )
    # contact_probs = torch.from_numpy(coords_hdf5[sample]['contact_probs'][:]).to(dtype=torch.float32) #(atoms_num, )
    # pae = torch.from_numpy(coords_hdf5[sample]['pae'][:]).to(dtype=torch.float32) #(L,L)

    f1d_esm2_650M = torch.from_numpy(fitness_h5[sample]['esm2_650M'][:]).to(dtype=torch.float32) #(L, 1280)
    esm1v_single_logits = torch.from_numpy(fitness_h5[sample]['esm1v_single_logits'][:]).to(dtype=torch.float32) #(5, L, 20)
    msa_embed = torch.from_numpy(fitness_h5[sample]["msa_transformer"][:]).to(dtype=torch.float32)  # [1, L, 768]
    # msa_count_matrix = torch.from_numpy(fitness_h5[sample]["msa_count_matrix"][:]).to(dtype=torch.float32)  # [1, L, 21]

    edge_index = build_distance_limited_knn_graph(atom_coords, k=FLAGS.k_neighbors, max_distance=8.0)  # (2, E)
    
    if 'ddG' in fitness_h5[sample]:
        ddG = torch.tensor(fitness_h5[sample]['ddG'][()]).to(dtype=torch.float32) #(L, 1)
    elif "ddG_label" in fitness_h5[sample]:
        ddG = torch.tensor(fitness_h5[sample]['ddG_label'][()]).to(dtype=torch.float32) #(L, 1)
    else:
        ddG = None
    if 'dTm' in fitness_h5[sample]:
        dTm = torch.tensor(fitness_h5[sample]['dTm'][()]).to(dtype=torch.float32) #(L, 1)
    else:
        dTm = None

    out_data = {
    "seq_str": seq_str[0][1],
    "target_tokens": target_tokens.to(device1),
    "atom_coords": atom_coords.to(device1), #.unsqueeze(0)
    "atom_types": atom_types.to(device1),
    "atom_pLDDT": atom_pLDDT.to(device1),
    "residue_types": residue_types.to(device1),
    "residue_mapping": residue_mapping.to(device1),
    "msa_embed": msa_embed.to(device1), # [L, 768]
    "f1d_esm2_650M": f1d_esm2_650M.to(device1), #(L, 1280)
    "esm1v_single_logits": esm1v_single_logits.to(device1), #(5, L, 20)
    "edge_index": edge_index.to(device1),  # (2, E)
    "ddG": ddG.to(device2) if ddG is not None else None,
    "dTm": dTm.to(device2) if dTm is not None else None
    }

    # print("f1d_esm2_650M", f1d_esm2_650M.shape, "esm1v_single_logits", esm1v_single_logits.shape)

    s1 = time.time()
    t_label = s1-s0
    # print("load feature", round(t_label,3))

    return out_data


def thread_Train_valid_stability(model, mut_data, wild_data, FLAGS):
    L1_loss = nn.L1Loss()
    label_ddG, label_dTm = mut_data['ddG'], mut_data['dTm']

    s0 = time.time()
    pred_ddG, pred_dTm = model(mut_data, wild_data)  #(L, 1), (L, 1)
    s1 = time.time()

    # print("pred_ddG", pred_ddG.shape, "label_ddG", label_ddG.unsqueeze(0).shape)
    # print("pred_ddG", pred_ddG, "label_ddG", label_ddG)
    # calculate soft MSE loss
    if label_ddG is not None:
        label_ddG = label_ddG.to(pred_ddG.device)
        ddG_loss = L1_loss(pred_ddG, label_ddG.unsqueeze(0))
    else:
        ddG_loss = torch.tensor(0.0, device=pred_ddG.device)

    if label_dTm is not None:
        label_dTm = label_dTm.to(pred_dTm.device)
        dTm_loss = L1_loss(pred_dTm, label_dTm)
    else:
        dTm_loss = torch.tensor(0.0, device=pred_dTm.device)

    ## Calculate Fitness Loss
    s2 = time.time()
    t_train1 = s1 - s0
    t_train2 = s2 - s1
    print("threadTrain", "train time", round(t_train1,3), "loss time", round(t_train2,3))

    return (ddG_loss, dTm_loss, pred_ddG, pred_dTm, label_ddG, label_dTm)
    

def infer_test_stability(model, mut_data, wild_data, FLAGS):

    label_ddG, label_dTm = mut_data['ddG'], mut_data['dTm']

    pred_ddG, pred_dTm = model(mut_data, wild_data)  #(L, 1), (L, 1)

    return pred_ddG

# write loss data into xls format
def write_loss_table(data_dict, output_dir, file_name):
    Loss_df = pd.DataFrame.from_dict(data_dict)
    Loss_table_path = os.path.join(output_dir, file_name)
    Loss_df.to_csv(Loss_table_path, sep="\t", na_rep='nan', index=False)


def thread_Train_valid_stability_reverse(model, mut_data, wild_data, FLAGS):
    # mse_loss = nn.MSELoss()
    L1_loss = nn.L1Loss()
    label_ddG, label_dTm = mut_data['ddG'], mut_data['dTm']
    label_ddG = -1 * label_ddG

    s0 = time.time()
    pred_ddG, pred_dTm = model(wild_data, mut_data)  #(L, 1), (L, 1)
    s1 = time.time()

    # print("pred_ddG", pred_ddG.shape, "label_ddG", label_ddG.unsqueeze(0).shape)
    print("pred_ddG", pred_ddG, "label_ddG", label_ddG)
    # calculate soft MSE loss
    if label_ddG is not None:
        label_ddG = label_ddG.to(pred_ddG.device)
        ddG_loss = L1_loss(pred_ddG, label_ddG.unsqueeze(0))
    else:
        ddG_loss = torch.tensor(0.0, device=pred_ddG.device)

    if label_dTm is not None:
        label_dTm = label_dTm.to(pred_dTm.device)
        dTm_loss = L1_loss(pred_dTm, label_dTm)
    else:
        dTm_loss = torch.tensor(0.0, device=pred_dTm.device)

    ## Calculate Fitness Loss
    s2 = time.time()
    t_train1 = s1 - s0
    t_train2 = s2 - s1
    print("threadTrain", "train time", round(t_train1,3), "loss time", round(t_train2,3))

    return (ddG_loss, dTm_loss, pred_ddG, pred_dTm, label_ddG, label_dTm)