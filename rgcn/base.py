import os
import sys
import yaml
import csv
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
from torch_utils import train_torch
#from river_dl.mypreproc_utils import prep_all_data
#from river_dl.mypreproc_utils import reduce_training_data_random
from torch_utils import rmse_masked as rmse_masked
#from evaluate import combined_metrics
from model import RGCN_v1 as Model
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
model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, device=device, seed=42)
opt = optim.Adam(model.parameters(), lr=0.01)
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
            device=device)

# predict
# data = np.load(os.path.join(data_processing_dir, 'data', 'prepped.npz'), allow_pickle=True)
# in_dim = len(data['x_vars'])
# adj_mx = data['dist_matrix']
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = Model(input_dim=in_dim, hidden_dim=config['hidden_size'], adj_matrix=adj_mx, device=device, seed=42)
model.load_state_dict(torch.load(f"{outdir}/finetuned_weights.pth"))
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
                         time_idx_name="date")


# def get_grp_arg(metric_type):
#     if metric_type == 'overall':
#         return None
#     elif metric_type == 'month':
#         return 'month'
#     elif metric_type == 'reach':
#         return 'seg_id_nat'
#     elif metric_type == 'month_reach':
#         return ['seg_id_nat', 'month']

# metric_types = ['overall', 'month', 'reach', 'month_reach']

# for metric_type in metric_types:
#     grp_arg = get_grp_arg(metric_type)
#     outfile = f"{outdir}/metrics/metrics.csv"
#     ensure_dir(outfile)
#     combined_metrics(obs_file='../../datasets/colorado/All_Reservoirs_Combined.csv',
#                      pred_trn=f"{outdir}/trn_preds.feather",
#                      pred_val=f"{outdir}/val_preds.feather",
#                      pred_tst=f"{outdir}/tst_preds.feather",
#                      group_spatially=False if not grp_arg else True if "COMID" in grp_arg else False,
#                      group_temporally=False if not grp_arg else 'M' if "month" in grp_arg else False,
#                      outfile=outfile,
#                      spatial_idx_name="COMID")
