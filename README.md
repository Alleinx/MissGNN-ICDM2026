## Anonymous ICDM 2026 Submission

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