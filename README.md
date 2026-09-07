# [ICDM 2026] MissGNN: Missingness-Guided Graph Neural Network for Feature- and Sample-Aware Imputation

### Introduction
Missing values are a common problem that poses significant challenges to data analysis and machine learning. This problem necessitates the development of an effective imputation method to fill in the missing values accurately, thereby enhancing the overall quality and utility of the datasets. 

In this work, we propose MissGNN, a novel imputation method that explicitly leverages missingness information and both feature- and sample-level correlations through carefully designed masking schemes. MissGNN first models the data as a bipartite graph and uses a graph neural network to learn node embeddings, where a novel embedding initialization process directly incorporates the missingness information. These embeddings are then optimized through MissGNN's novel feature correlation unit FCU and sample correlation unit SCU, which effectively capture feature and sample correlations for imputation. 

Experimental results on 8 benchmark datasets against 12 popular and state-of-the-art imputation baselines under three different missingness settings (MCAR, MAR, and MNAR) demonstrate the effectiveness of MissGNN, achieving the best average MAE scores on 7 datasets and the second-best on the remaining one.

## Requirements & Setup

We conduct our experiment on a server with the following environments:

- Ubuntu `22.04`
- CUDA `12.1`
- conda `23.9.0`
- Python `3.11.6`
- Torch `2.1.0` & Torchvision `0.16.0` (Usually need to install manually to make sure the library can utilize GPU)

After you prepare your environments, you can install other requirements:

```setup
pip install -r requirements.txt
```

Finally, you need to install `pytorch scatter`. Here are the install command for our environment, you can refer to the documentation and your settings to select the install version.

```setup
# GPU
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# CPU
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Dataset

All datasets should be placed inside the `uci/raw_data` folder.

Expected folder structure:

```
├── uci
│   ├── __init__.py
│   ├── raw_data
│   │   ├── cmc
│   │   ├── concrete
│   │   ├── german
│   │   ├── housing
│   │   ├── ionosphere
│   │   ├── power
│   │   ├── steel
│   │   └── yacht
│   ├── uci_data.py
│   └── uci_subparser.py
```


## Training & Evaluation

We provide the startup parameters used in out experiment, and all options and parameters are specified in the `.sh` file in root folder. For more training options, look at the arguments in `train_mdi.py` and `uci/uci_subparser.py`.

### 1. Experiment 1

> Imputation under different simulated missingness scenarios.

```train
bash run_exp1_impute.sh       # MCAR
bash run_exp1_impute_mar.sh   # MAR
bash run_exp1_impute_mnar.sh  # MNAR
```

### 2. Experiment 2

> Robustness against various ratios of missingness.

```train
bash run_exp2_robust.sh
```

### 3. Experiment 3

> Ablation study and hyperparameter explore

```train
bash run_exp3_ablation.sh
```

### 4. Baselines

```train
python baseline_mdi.py --method mean uci --train_edge 0.7 --data yacht
```
