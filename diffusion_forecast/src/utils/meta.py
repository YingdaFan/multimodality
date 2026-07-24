"""
Sidecar metadata I/O for forecast pipeline.

训练阶段 (forecast_loader) 在产生 tst.npy 的同一目录写一份 meta.json，
记录该次训练实际用到的 stride/window/pred_len/n_basins 等。
后处理阶段 (postprocess_forecast) 优先读 meta.json，避免 shell flag 与
训练实际值脱钩。
"""
import json
import os

META_FILENAME = 'meta.json'


def write_meta(pred_dir, **kwargs):
    """Write meta.json into pred_dir. Overwrites if exists."""
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, META_FILENAME)
    with open(path, 'w') as f:
        json.dump(kwargs, f, indent=2, default=str)
    return path


def read_meta(pred_dir):
    """Return dict from meta.json, or None if not present."""
    path = os.path.join(pred_dir, META_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
