
您需要准备的输入文件：
1. AlphaFold3的预测结构：00_wild_type_AF3_pred
2. 突变蛋白信息文件：00_mutations_info_examples.csv
    其中有四列：protein(野生型蛋白名称)   mut_info(突变信息)   mut_seq(突变蛋白序列)   wt_seq(野生型蛋白质序列)
     mut_info:  H48Q表示野生型第28位残基H突变成Q

脚本运行顺序：
1. 获取序列的语言模型embedding输入文件hdf5:          01_get_embedding.sh
2. 获取突变蛋白与野生型蛋白质结构输入文件hdf5：      02_get_structures.sh
3. 运行FA-stab:                                                          03_run_FA-stab.sh
4. ddG结果预测文件：                                                Test_Stability_results.xls

FA-stab的脚本与参数位置：/lustre/home/yhchen/protein_stability/scripts/FA-stab

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


