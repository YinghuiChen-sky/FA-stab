import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.math import soft_one_hot_linspace
from e3nn.o3 import FullyConnectedTensorProduct, Linear

seq2token_dict = {'A':0, 'R':1, 'N':2, 'D':3, 'C':4, 'Q':5, 'E':6, 'G':7, 'H':8, 'I':9, 'L':10, 'K':11, 'M':12, 'F':13, 
                        'P':14, 'S':15, 'T':16, 'W':17, 'Y':18, 'V':19, '-':20, 'X':21} 

RESIDUE_MAP = {'ALA': 0, 'ARG':1, 'ASN':2, 'ASP':3, 'CYS':4, 'GLN':5, 'GLU':6, 'GLY':7, 'HIS':8, 'ILE':9, 'LEU':10, 
                'LYS':11, 'MET':12, 'PHE':13, 'PRO':14, 'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19, '-':20, 'UNK':21}

ATOM_TYPE_MAP = {'C': 0, 'N': 1, 'O': 2, 'S': 3}

# 蛋白质特征编码器
class ProteinFeatureEncoder(nn.Module):
    def __init__(self, plddt_dim=1):
        super().__init__()
        
        # 1. 原子嵌入
        self.atom_embedding = nn.Embedding(4, 8)
        nn.init.xavier_uniform_(self.atom_embedding.weight)
        
        # 2. 残基嵌入
        self.residue_embedding = nn.Embedding(21, 16)
        nn.init.xavier_uniform_(self.residue_embedding.weight)
        
        # 3. pLDDT处理
        self.plddt_encoder = nn.Sequential(
            nn.Linear(plddt_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 8))
        
        # 输出为标量表示
        self.irreps_out = o3.Irreps(f"32x0e")
    
    def forward(self, atom_types, residue_types, plddt):
        # 原子嵌入
        atom_emb = self.atom_embedding(atom_types)
        
        # 残基嵌入
        residue_emb = self.residue_embedding(residue_types)
        # print("residue_emb", residue_emb.shape)
        
        # pLDDT嵌入
        plddt_emb = self.plddt_encoder(plddt)
        # print("plddt_emb", plddt_emb.shape)
        
        # 合并特征 (按原子)
        atom_features = torch.cat([
            atom_emb,
            residue_emb,
            plddt_emb,
        ], dim=-1)
        
        return atom_features

# 等变卷积层
class EquivariantConvolution(nn.Module):
    def __init__(self, irreps_in, out_channels, max_sh_order=2, num_radial_basis=11, radial_hidden_dim=12):
        super().__init__()
        self.max_sh_order = max_sh_order
        self.num_radial_basis = num_radial_basis
        
        # 输入和输出的不可约表示
        self.irreps_in = irreps_in
        self.irreps_sh = o3.Irreps.spherical_harmonics(max_sh_order)
        
        # 输出不可约表示 (l=0,1,2)
        self.irreps_out = o3.Irreps([
            (out_channels, (l, (-1)**l)) 
            for l in range(max_sh_order + 1)
        ]).simplify()
        
        # 张量积层
        self.tp = FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False
        )
        
        # 径向网络
        self.radial_net = FullyConnectedNet(
            [num_radial_basis, radial_hidden_dim, self.tp.weight_numel],
            act=nn.ReLU()
        )
        
        # 球谐函数计算器
        self.sh = o3.SphericalHarmonics(
            list(range(max_sh_order + 1)), 
            normalize=True, 
            normalization='component'
        )

    def forward(self, pos, features, edge_index):
        src, dst = edge_index
        vec = pos[src] - pos[dst]
        distances = torch.norm(vec, dim=1)
        
        # 计算径向基
        radial_basis = soft_one_hot_linspace(
            distances, 
            start=0.0, 
            end=8.0, 
            number=self.num_radial_basis,
            basis='gaussian',
            cutoff=False
        )
        
        # 径向网络输出权重
        weight = self.radial_net(radial_basis)
        
        # 计算球谐函数
        directions = F.normalize(vec, dim=1)
        sh = self.sh(directions)
        
        # 执行张量积操作
        out = self.tp(
            features[dst],  # 输入特征
            sh,             # 球谐函数
            weight          # 径向网络生成的权重
        )
        
        # 聚合邻居贡献
        output = torch.zeros(features.size(0), self.irreps_out.dim, device=features.device)
        src_indices = src.unsqueeze(1).expand(-1, self.irreps_out.dim)
        output.scatter_add_(0, src_indices, out)
        
        return output

# 点归一化层
class PointwiseNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        norm = torch.norm(x, dim=1, keepdim=True)
        return x / (norm + 1e-6)

# 点非线性激活
class PointwiseNonlinearity(nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        self.irreps_in = irreps_in
        self.scalar_act = nn.Softplus()
        
        # 计算最大角量子数
        self.lmax = max(ir.l for (mul, ir) in irreps_in) if irreps_in else 0
        
        # 为每个角动量阶创建可学习的偏置
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(self.lmax + 1)
        ])
    
    def forward(self, x):
        output = []
        start_idx = 0
        
        # 处理每个不可约表示块
        for (mul, ir) in self.irreps_in:
            dim = ir.dim
            block = x[:, start_idx:start_idx + mul * dim]
            start_idx += mul * dim
            
            if ir.l == 0:  # 标量部分
                # 直接应用激活函数
                activated = self.scalar_act(block + self.biases[0])
                output.append(activated)
            else:  # 向量/张量部分
                # 重塑为 [num_nodes, mul, dim]
                block = block.view(-1, mul, dim)
                
                # 计算范数
                norms = torch.norm(block, dim=2)  # [num_nodes, mul]
                
                # 应用激活函数到范数
                gate = self.scalar_act(norms + self.biases[ir.l])
                
                # 应用门控
                activated = block * gate.unsqueeze(2)
                output.append(activated.view(-1, mul * dim))
        
        return torch.cat(output, dim=1)

# 自相互作用层
class SelfInteraction(nn.Module):
    def __init__(self, irreps_in, out_channels):
        super().__init__()
        # 构建输出不可约表示
        irreps_out = []
        for mul, ir in irreps_in:
            irreps_out.append((out_channels, ir))
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        
        # 线性变换层
        self.linear = Linear(irreps_in, self.irreps_out)
    
    def forward(self, x):
        return self.linear(x)

# 手动提取标量部分的辅助函数
def get_scalar_indices(irreps):
    """获取标量部分在特征向量中的索引"""
    start_idx = 0
    scalar_indices = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:  # 标量部分 (偶宇称)
            end_idx = start_idx + mul * ir.dim
            scalar_indices.extend(range(start_idx, end_idx))
        start_idx += mul * ir.dim
    return scalar_indices

# 等变主干网络
class EquivariantBackbone(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        
        # 初始自相互作用层
        self.initial_si = SelfInteraction(o3.Irreps(f"{feature_dim}x0e"), 128)
        current_irreps = self.initial_si.irreps_out
        
        # 等变卷积层和中间层
        self.equiv_convs = nn.ModuleList()
        self.pointwise_norms = nn.ModuleList()
        self.intermediate_si = nn.ModuleList()
        self.pointwise_nonlin = nn.ModuleList()
        
        # 三个主要块
        block_configs = [
            (128, 128),
            (128, 64),
            (64, 32)
        ]
        
        for in_channels, out_channels in block_configs:
            # 等变卷积层
            conv = EquivariantConvolution(current_irreps, out_channels, max_sh_order=2)
            self.equiv_convs.append(conv)
            current_irreps = conv.irreps_out
            
            # 点归一化层
            self.pointwise_norms.append(PointwiseNorm())
            
            # 自相互作用层
            si = SelfInteraction(current_irreps, out_channels)
            self.intermediate_si.append(si)
            current_irreps = si.irreps_out
            
            # 点非线性层
            self.pointwise_nonlin.append(PointwiseNonlinearity(current_irreps))
        
        # 最终自相互作用层
        self.final_si = SelfInteraction(current_irreps, 32)
        current_irreps = self.final_si.irreps_out
        
        # 获取标量部分的索引
        self.scalar_indices = get_scalar_indices(current_irreps)
        self.scalar_dim = len(self.scalar_indices)
        
        if self.scalar_dim == 0:
            self.scalar_dim = 32
            print("Warning: No scalar features found, setting scalar_dim to 32")
    
    def forward(self, pos, features, edge_index):
        # 初始自相互作用
        x = self.initial_si(features)
        
        # 三个主要块
        for i in range(3):
            # 等变卷积
            x = self.equiv_convs[i](pos, x, edge_index)
            
            # 点归一化
            x = self.pointwise_norms[i](x)
            
            # 自相互作用
            x = self.intermediate_si[i](x)
            
            # 点非线性
            x = self.pointwise_nonlin[i](x)
        
        # 最终自相互作用
        x = self.final_si(x)
        
        # 只保留标量部分
        x = x[:, self.scalar_indices]
        
        return x

# 残基特征聚合器
class ResidueFeatureAggregator(nn.Module):
    def __init__(self, atom_dim, residue_dim=20):
        super().__init__()
        self.residue_dim = residue_dim
        self.atom_dim = atom_dim
        
        # 残基级特征转换
        self.residue_transformer = nn.Sequential(
            nn.Linear(atom_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, residue_dim))
    
    def forward(self, atom_features, residue_mapping):
        """
        atom_features: [num_atoms, atom_dim]
        residue_mapping: [num_atoms] 每个原子所属残基的索引
        """
        # 聚合残基特征 (平均池化)
        residue_features = torch.zeros(
            residue_mapping.max() + 1, 
            self.atom_dim,
            device=atom_features.device
        )
        residue_features.scatter_add_(0, residue_mapping.unsqueeze(1).expand(-1, self.atom_dim), atom_features)
        # print("residue_features", residue_features.shape)

        # 计算每个残基的原子数量
        residue_counts = torch.zeros_like(residue_features[:, 0])
        residue_counts.scatter_add_(0, residue_mapping, torch.ones_like(residue_mapping, dtype=torch.float))
        # print("residue_counts", residue_counts.shape)
        
        # 平均特征
        residue_features = residue_features / residue_counts.unsqueeze(1)
        
        # 转换为残基级表示
        return self.residue_transformer(residue_features)

# 完整的蛋白质fitness预测模型
class ProteinFitnessPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. 特征编码器
        self.feature_encoder = ProteinFeatureEncoder(plddt_dim=1)
        
        # 2. 等变主干网络
        self.equiv_backbone = EquivariantBackbone(self.feature_encoder.irreps_out.dim)
        
        # 3. 残基特征聚合器
        self.residue_aggregator = ResidueFeatureAggregator(
            self.equiv_backbone.scalar_dim,
            residue_dim=20
        )
        
        # 4. MSA Transformer Embedding 处理
        self.msa_encoder = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 20))
        
        # 5. ESM_650M Embedding 处理
        self.ESM_650M_encoder = nn.Sequential(
            nn.LayerNorm(1280),
            nn.Linear(1280, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 20)
        )
        
        
    
    def forward(self, data):
        ## atoms data
        atom_coords = data['atom_coords'] # 原子坐标 [num_atoms, 3]
        atom_types = data['atom_types']  # 原子类型 [num_atoms]
        atom_pLDDT = data['atom_pLDDT'].unsqueeze(-1)   # pLDDT置信度 [num_residues, 1]
        ## residues data
        residue_types = data['residue_types']  # 残基类型 [num_atoms]
        residue_mapping = data['residue_mapping']  # 原子到残基的映射 [num_atoms]

        ## LLM embeddings
        msa_embeddings = data['msa_embed']  # MSA嵌入 [num_residues, msa_dim=768]
        f1d_esm2_650M = data['f1d_esm2_650M']  # ESM-2单体特征 [num_residues, 1280]
        esm1v_single_logits = data['esm1v_single_logits']  # ESM-1v单体logits #(1, 5, L, 20)
        # esm1v_double_logits = data['esm1v_double_logits']  # ESM-1v双体logits  #(1, 5, L, L 400)

        # 特征编码
        atom_features = self.feature_encoder(
            atom_types, 
            residue_types, 
            atom_pLDDT,  
        )
        
        # 等变处理
        print("atom_features", atom_features.shape, "atom_coords", atom_coords.shape, "edge_index", data['edge_index'].shape)
        equiv_features = self.equiv_backbone(
            atom_coords, 
            atom_features, 
            data['edge_index']
        )
        
        print("equiv_features", equiv_features.shape, "residue_mapping", residue_mapping.shape)
        # 聚合到残基级
        structure_features = self.residue_aggregator(equiv_features, residue_mapping)
        structure_features = structure_features[None, None, ...]  # [1, 1, L, 20]
        # print("structure_features", structure_features.shape)
        
        # MSA embedding
        msa_embeddings = self.msa_encoder(msa_embeddings)[None, None, ...] #[1, 1, L, 20]

        # ESM_650M Embedding 处理
        esm_650M_embed = self.ESM_650M_encoder(f1d_esm2_650M)[None, None, ...] #[1, 1, L, 20]
        
        # ESM1v logits shape调整
        esm1v_single_logits = esm1v_single_logits[None,...] #[1, 5, L, 20]
        
        print("structure_features", structure_features.shape, "msa_embeddings", msa_embeddings.shape, "esm_650M_embed", esm_650M_embed.shape, "esm1v_single_logits", esm1v_single_logits.shape)

        # single_logits, single_feat = self.GAT_Fitness(structure_features, combined_feat_2d, esm1v_single_logits, msa_embeddings, esm_650M_embed)
        
        return structure_features, esm_650M_embed, msa_embeddings, esm1v_single_logits


class Stability_Predict(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.Fitness_model = ProteinFitnessPredictor()
        # Embedding 混合权重
        self.logits_coef1 = torch.nn.Parameter(torch.tensor([0.5, 0.06, 0.06, 0.06, 0.06, 0.06, 0.1, 0.1], requires_grad = True))
        self.fitness_logits_dim = 20
        self.mlp_for_ddG = torch.nn.Sequential(torch.nn.LayerNorm(self.fitness_logits_dim), torch.nn.Linear(self.fitness_logits_dim, self.fitness_logits_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(self.fitness_logits_dim * 2, 1))
        self.mlp_for_dTm = torch.nn.Sequential(torch.nn.LayerNorm(self.fitness_logits_dim), torch.nn.Linear(self.fitness_logits_dim, self.fitness_logits_dim * 2), torch.nn.LeakyReLU(), torch.nn.Linear(self.fitness_logits_dim * 2, 1))
        self.finetune_ddG_coef = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        self.finetune_dTm_coef = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
        
    def forward(self, mut_data, wt_data):
        wt_struct_feat, wt_esm_650M_embed, wt_msa_embed, wt_esm1v_single_logits = self.Fitness_model(wt_data)  # wt_embedding=(batch=1, L, 32)
        mut_struct_feat, mut_esm_650M_embed, mut_msa_embed, mut_esm1v_single_logits = self.Fitness_model(mut_data)

        d_struct_feat = mut_struct_feat - wt_struct_feat
        d_esm_650M_embed = mut_esm_650M_embed - wt_esm_650M_embed
        d_msa_embed = mut_msa_embed - wt_msa_embed
        d_esm1v_single_logits = mut_esm1v_single_logits - wt_esm1v_single_logits

        mut_logits = self.logits_coef1[None, :, None, None] * torch.cat((d_struct_feat, d_esm1v_single_logits, d_esm_650M_embed, d_msa_embed), dim = 1)
        mut_logits = mut_logits.mean(1) # [batch=1, L, 20]
        # print("mut_logits", mut_logits.shape)
        # print("logits_coef1", self.logits_coef1)

        # identify mutation position
        wt_tokens = wt_data["target_tokens"]
        mut_tokens = mut_data["target_tokens"]
        print("wt_tokens", wt_tokens.shape, "mut_tokens", mut_tokens.shape)
        mut_pos = (wt_tokens != mut_tokens).float().unsqueeze(-1)  # [batch=1, L, 1]
        # print("mut_pos", mut_pos.shape)

        ddG_logits = self.mlp_for_ddG((mut_logits * mut_pos).sum(1))

        ddG = ddG_logits.squeeze(-1) * self.finetune_ddG_coef

        dTm_logits = self.mlp_for_dTm((mut_logits * mut_pos).sum(1))
        dTm = dTm_logits.squeeze(-1) * self.finetune_dTm_coef
        
        # print("ddG", ddG, "ddG_coef", self.finetune_ddG_coef)
        # print("dTm", dTm, "dTm_coef", self.finetune_dTm_coef)
        
        return ddG, dTm