"""
End-to-end runner for the CoupledFMDiff Stage-1 model.

Standalone analogue of `lstm/base.py` — reads `data_processing/data/prepped.npz`,
trains the coupled FM + Diffusion model with NaN-mask loss, and writes
`{trn,val,tst}.npy` predictions plus a per-epoch component log.

Usage:
    cd lstm/Model/flowmatchingdiffusion
    python -u base.py                    # use defaults below
    CUDA_VISIBLE_DEVICES=1 python -u base.py # pick a specific GPU

Outputs (under <outdir>):
    finetuned_weights.pth   best-val checkpoint
    train_log.csv           per-epoch component losses
    preds/trn.npy           predictions on training partition
    preds/val.npy           predictions on validation partition
    preds/tst.npy           predictions on test partition

This file is deliberately self-contained: it only imports from this
sub-package and from the prepped.npz produced by the upstream preprocess
script. Nothing in `lstm/` parent is touched.
"""

import os
import sys
import yaml
import random

# Ensure the parent `lstm/` directory is on the path so that
# `from Model.flowmatchingdiffusion import ...` resolves when this file
# is run directly. (Same pattern as lstm/base.py.)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LSTM_DIR = os.path.dirname(os.path.dirname(_THIS_DIR))  # .../lstm/
_IMPUTATION_DIR = os.path.dirname(_LSTM_DIR)             # .../imputation/
_DATA_PROCESSING_DIR = os.path.join(_IMPUTATION_DIR, 'data_processing')
sys.path.insert(0, _LSTM_DIR)

import numpy as np
import torch
import torch.optim as optim

from Model.flowmatchingdiffusion import (
    CoupledFMDiff,
    train_torch_fmdiff,
    predict_fmdiff_from_io_data,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42

def _set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Sensible defaults; override by editing here or by writing a YAML file
# named `config_fmdiff.yml` in this directory.
DEFAULT_CONFIG = {
    # Backbone selection (controls the score / vel network architecture)
    #   'dlinear'     — DLinear + FiLM (lightweight, ~150K params)
    #   'transformer' — DiT-style Transformer (~2.7M params, expressive)
    'backbone': 'dlinear',
    'backbone_kwargs': None,      # None -> use defaults for the chosen backbone

    # DLinear-specific (used only when backbone='dlinear' and
    # backbone_kwargs is None)
    'kernel_size': 25,
    'time_embed_dim': 64,
    'individual': False,

    # Generic
    'share_backbone': False,
    'dropout': 0.1,
    'beta_min': 0.1,
    'beta_max': 20.0,

    # Training
    'ft_epochs': 30,              # max epochs
    'early_stopping': 20,
    'finetune_learning_rate': 1e-3,
    'weight_decay': 0.0,
    'grad_clip': 3.0,
    'shuffle': True,              # IMPORTANT: early windows often all-NaN
    'lambda_ode_max': 1.0,
    'lambda_ode_ramp_frac': 0.2,  # ramp 0 -> max over first 20% of epochs

    # Sampling for inference
    'sample_steps': 50,           # reverse-ODE integration steps
    'sample_method': 'vel',       # 'vel' | 'score' | 'avg'
    'n_samples_at_predict': 1,    # >1 = ensemble mean

    # Optional slow eval during training (epochs; 0 = off)
    'sample_eval_every': 0,
}

CFG_PATH = os.path.join(_THIS_DIR, 'config_fmdiff.yml')
if os.path.exists(CFG_PATH):
    with open(CFG_PATH) as f:
        DEFAULT_CONFIG.update(yaml.safe_load(f) or {})

# Output directory: lstm/output_fmdiff_<backbone>/  (one per backbone, so
# back-to-back A/B runs do not overwrite each other).
OUT_DIR = os.environ.get(
    'FMDIFF_OUT_DIR',
    os.path.join(_LSTM_DIR, f'output_fmdiff_{DEFAULT_CONFIG["backbone"]}'),
)
PREDS_DIR = os.path.join(OUT_DIR, 'preds')
os.makedirs(PREDS_DIR, exist_ok=True)

WEIGHTS_FILE = os.path.join(OUT_DIR, 'finetuned_weights.pth')
LOG_FILE = os.path.join(OUT_DIR, 'train_log.csv')

# Device
DEVICE = 'cuda'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _set_global_seed(GLOBAL_SEED)
    print(f'[INFO] Global random seed set to {GLOBAL_SEED}')

    # ---- 1. Load prepped.npz ----
    npz_path = os.path.join(_DATA_PROCESSING_DIR, 'data', 'prepped.npz')
    print(f'[INFO] loading {npz_path}')
    data = np.load(npz_path, allow_pickle=True)

    x_trn = data['x_trn']
    y_trn = data['y_obs_trn']
    x_val = data['x_val'] if 'x_val' in data.files else None
    y_val = data['y_obs_val'] if 'y_obs_val' in data.files else None
    x_tst = data['x_tst'] if 'x_tst' in data.files else None
    y_tst = data['y_obs_tst'] if 'y_obs_tst' in data.files else None

    n_segs = len(np.unique(data['ids_trn']))
    in_dim = len(data['x_vars'])
    seq_len = x_trn.shape[1]
    print(f'[INFO] n_segs={n_segs}  in_dim={in_dim}  seq_len={seq_len}')
    print(f'[INFO] x_trn={x_trn.shape}, y_obs_trn nan_frac={np.isnan(y_trn).mean():.4f}')

    # ---- 2. Build model ----
    cfg = DEFAULT_CONFIG
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device = {device}')
    model = CoupledFMDiff(
        input_dim=in_dim,
        seq_len=seq_len,
        backbone=cfg['backbone'],
        backbone_kwargs=cfg.get('backbone_kwargs'),
        kernel_size=cfg['kernel_size'],
        time_embed_dim=cfg['time_embed_dim'],
        individual=cfg['individual'],
        share_backbone=cfg['share_backbone'],
        dropout=cfg['dropout'],
        beta_min=cfg['beta_min'],
        beta_max=cfg['beta_max'],
        sample_steps=cfg['sample_steps'],
        sample_method=cfg['sample_method'],
        seed=GLOBAL_SEED,
    ).to(device)
    print(f'[INFO] backbone = {cfg["backbone"]}')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[INFO] model params: {n_params:,}')

    # ---- 3. Optimizer ----
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg['finetune_learning_rate'],
        weight_decay=cfg['weight_decay'],
    )

    # ---- 4. Train ----
    train_torch_fmdiff(
        model=model,
        optimizer=opt,
        x_train=x_trn, y_train=y_trn,
        x_val=x_val, y_val=y_val,
        x_tst=x_tst, y_tst=y_tst,
        batch_size=n_segs,
        max_epochs=cfg['ft_epochs'],
        early_stopping_patience=cfg['early_stopping'],
        shuffle=cfg['shuffle'],
        weights_file=WEIGHTS_FILE,
        log_file=LOG_FILE,
        device=device,
        grad_clip=cfg['grad_clip'],
        lambda_ode_max=cfg['lambda_ode_max'],
        lambda_ode_ramp_frac=cfg['lambda_ode_ramp_frac'],
        sample_eval_every=cfg['sample_eval_every'],
    )

    # ---- 5. Reload best weights ----
    if os.path.exists(WEIGHTS_FILE):
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
        print(f'[INFO] loaded best checkpoint from {WEIGHTS_FILE}')

    # ---- 6. Predict on all partitions ----
    for partition in ['trn', 'val', 'tst']:
        if f'x_{partition}' not in data.files:
            continue
        outfile = os.path.join(PREDS_DIR, partition)
        predict_fmdiff_from_io_data(
            model=model,
            io_data=data,
            partition=partition,
            outfile=outfile,
            num_steps=cfg['sample_steps'],
            method=cfg['sample_method'],
            n_samples=cfg['n_samples_at_predict'],
            batch_size=n_segs,
            device=device,
        )

    print(f'\n[DONE] outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()
