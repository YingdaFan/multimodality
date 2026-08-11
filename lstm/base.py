import os
import sys
import yaml
import csv
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
imputation_dir = os.path.dirname(current_dir)
data_processing_dir = os.path.join(imputation_dir, 'data_processing')

config_file_path = os.path.join(current_dir, 'config.yml')
with open(config_file_path, 'r') as f:
    config = yaml.safe_load(f)
sys.path.insert(0, data_processing_dir)

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import importlib
import sys

# ============================================
# Fix all random seeds for reproducibility
# ============================================
GLOBAL_SEED = 42

def set_global_seed(seed=GLOBAL_SEED):
    """Set all random seeds to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seed before any other operations
set_global_seed(GLOBAL_SEED)
print(f"[INFO] Global random seed set to {GLOBAL_SEED}")
from torch_utils import train_torch
#from river_dl.mypreproc_utils import prep_all_data
#from river_dl.mypreproc_utils import reduce_training_data_random
from torch_utils import rmse_masked as rmse_masked
#from evaluate import combined_metrics
from model import LSTM as Model
from predict import predict_from_io_data

# outdir = './output'
outdir = os.path.join(current_dir, 'output')

# CUDA device configuration (use environment variable or default to cuda:0)
cuda_device = 'cuda'

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)




data = np.load(os.path.join(data_processing_dir, 'data', 'prepped.npz'), allow_pickle=True)
num_segs = len(np.unique(data['ids_trn']))
adj_mx = data['dist_matrix']
in_dim = len(data['x_vars'])
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
dropout = config.get('dropout', 0.0)

use_smap = bool(config.get('use_smap', False)) or os.environ.get('USE_SMAP') == '1'
smap_provider = None
if use_smap:
    from smap_encoder import LSTMWithSMAP
    from smap_data import SMAPProvider
    smap_provider = SMAPProvider(os.path.join(data_processing_dir, 'data', 'smap_packed.npz'))
    for split in ['trn', 'val', 'tst']:
        smap_provider.register_split(split, data[f'ids_{split}'], data[f'times_{split}'])
    print(f"[INFO] SMAP modality ON (d_smap={config.get('d_smap', 32)})")

def build_model():
    if use_smap:
        return LSTMWithSMAP(input_dim=in_dim, hidden_dim=config['hidden_size'],
                            d_smap=config.get('d_smap', 32), adj_matrix=adj_mx,
                            dropout=dropout, device=device, seed=42)
    return Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, dropout=dropout, device=device, seed=42)

model = build_model()
weight_decay = config.get('weight_decay', 0.0)  # default 0.0 (no regularization)
opt = optim.Adam(model.parameters(), lr=config['finetune_learning_rate'], weight_decay=weight_decay)
# if 'pre_train' in locals():
#     model.load_state_dict(torch.load(f"../{outdir}/pretrained_weights.pth", map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))

train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            x_tst=data['x_tst'],
            y_tst=data['y_obs_tst'],
            max_epochs=config['ft_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size=num_segs,
            weights_file=f"{outdir}/finetuned_weights.pth",
            log_file=f"{outdir}/train_log.csv",
            device=device,
            smap_provider=smap_provider)

# predict
data = np.load(os.path.join(data_processing_dir, 'data', 'prepped.npz'), allow_pickle=True)
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
model = torch.load(f"{outdir}/finetuned_weights.pth", map_location=device, weights_only=False)
partitions = ['trn','val','tst']


for partition in partitions:
    outfile = f"{outdir}/preds/{partition}.npy"
    ensure_dir(outfile)
    predict_from_io_data(model=model,
                         io_data=os.path.join(data_processing_dir, 'data', 'prepped.npz'),
                         partition=partition,
                         outfile=outfile,
                         log_vars=False,
                         trn_offset=1,
                         tst_val_offset=1,
                         spatial_idx_name="spatial_idx_name",
                         time_idx_name="date",
                         smap_provider=smap_provider)


