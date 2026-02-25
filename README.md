<h1 align="center">FA-stab</h1>
![FA-stab](figures/FA-stab_Model_Architecture.svg)

## Overview
FA-stab is an all-atom-aware model for predicting changes in protein stability (∆∆G). FA-stab employs Euclidean Neural Networks (e3nn) to capture the three-dimensional positions of all atoms, including side-chain atoms, and further augments its predictions with evolutionary information derived from protein language models. 

## Installation
1. Download `FA-stab`

```bash
git clone https://github.com/YinghuiChen-sky/FA-stab.git
cd FA-stab
```

2. Install conda environment for `FA-stab`
```bash
conda env create -f environment.yml
conda activate FAstab
```

3. Install other dependent softwares
When preparing input files for FA-stab, the scripts require FoldX5 and hh-suite(hhfilter). You can install these softwares following download links belows.
```bash
Download  FoldX5:      http://foldxsuite.crg.es
Download hh-suite:     https://github.com/soedinglab/hh-suite
```
FA-stab needs parameters from ESM protein language models: ESM (650M), MSAtransformer, and ESM1v. 
Download  ESM2-650M, ESMIF and MSA Transformer checkpoints:
```bash
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm_msa1b_t12_100M_UR50S.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/regression/esm_msa1b_t12_100M_UR50S-contact-regression.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_2.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_3.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_4.pt
wget -c https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_5.pt
```

## Prepare raw input files for `FA-stab`
The raw input files you need to prepare:
(1)  Predicted structures from AlphaFold3: example/00_wild_type_AF3_pred
(2)  Mutation information file: example/00_mutations_info_examples.csv
        This file contains four columns: protein, mut_info, mut_seq, and wt_seq.
        mut_info follows the format, e.g., H48Q, which indicates that residue 48 in the wild‑type protein (originally H) is mutated to Q.

## Run `FA-stab`
Script execution order:

(1) Obtain HDF5 files of sequence embeddings from language models: 

```bash
sh 01_get_embedding.sh
```

(2) Obtain HDF5 files containing structures of both mutant and wild‑type proteins: 
```bash
sh 02_get_structures.sh
```

(3) Run FA-stab
```bash
03_run_FA-stab.sh
```

ddG prediction results file from FA-stab: FA-stab_prediction_results.xls


