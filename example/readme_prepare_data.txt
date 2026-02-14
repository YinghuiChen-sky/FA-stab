在准备FA-stab的输入文件时，脚本需要使用到FoldX5、ESM 蛋白质语言模型的参数:ESM(650M)、MSAtransformer、ESM1v。在本文最后列出下载链接。

您需要准备的原始输入文件：
1. AlphaFold3的预测结构：00_wild_type_AF3_pred
2. 突变蛋白信息文件：       00_mutations_info_examples.csv
    其中有四列：protein    mut_info    mut_seq    wt_seq
     mut_info:  H48Q表示野生型第28位残基H突变成Q

脚本运行顺序：
1. 获取序列的语言模型embedding输入文件hdf5:          01_get_embedding.sh
2. 获取突变蛋白与野生型蛋白质结构输入文件hdf5：      02_get_structures.sh
3. 运行FA-stab:                                                          03_run_FA-stab.sh
4. ddG结果预测文件：                                                FA-stab_prediction_results.xls


Download  FoldX5:      http://foldxsuite.crg.es

Download  ESM2-650M, ESMIF and MSA Transformer checkpoints:
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_2.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_3.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_4.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_5.pt


