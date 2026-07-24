"""
Flow-Matching + Diffusion sub-package.

Self-contained alternative to the LSTM Stage-1 model. Designed so the
whole subdirectory can later be lifted out as a standalone method.

Components
----------
  Model:
    - DLinearBackboneTS : DLinear backbone with FiLM time conditioning
    - CoupledFMDiff     : main wrapper (drop-in for `lstm.model.LSTM`)
    - VPSchedule        : VP-SDE noise schedule

  Training:
    - train_torch_fmdiff : top-level epoch loop using `compute_loss`
    - train_loop_fmdiff  : per-epoch train logic with component logging
    - val_loop_fmdiff    : per-epoch val logic
    - sample_rmse_fmdiff : optional slow eval (full reverse-ODE sample
                           on a val batch -> RMSE in y space)

  Inference:
    - predict_torch_fmdiff       : batched reverse-ODE sampling
    - predict_fmdiff             : LSTM-compatible wrapper, saves .npy
    - predict_fmdiff_from_io_data: top-level entry that reads partition
                                   data from prepped.npz

Loss interface
--------------
The wrapper exposes `compute_loss(x, y_true)` which returns the
multi-objective FM + score + ODE consistency loss with NaN masking
matching `rmse_masked`. The training utilities here invoke that method
directly. To use this model with the existing `lstm/torch_utils.py::
train_loop`, that loop would need to dispatch on `compute_loss` (we
deliberately avoid modifying the parent module).
"""

from .dlinear_ts import DLinearBackboneTS
from .transformer_backbone_ts import TransformerBackboneTS
from .coupled_fmdiff import CoupledFMDiff, VPSchedule

from .torch_utils import (
    train_torch_fmdiff,
    train_loop_fmdiff,
    val_loop_fmdiff,
    sample_rmse_fmdiff,
)

from .predict import (
    predict_torch_fmdiff,
    predict_fmdiff,
    predict_fmdiff_from_io_data,
)

__all__ = [
    # backbones
    'DLinearBackboneTS', 'TransformerBackboneTS',
    # model
    'CoupledFMDiff', 'VPSchedule',
    # training
    'train_torch_fmdiff', 'train_loop_fmdiff',
    'val_loop_fmdiff', 'sample_rmse_fmdiff',
    # inference
    'predict_torch_fmdiff', 'predict_fmdiff', 'predict_fmdiff_from_io_data',
]
