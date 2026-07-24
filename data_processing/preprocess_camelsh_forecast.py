"""
CAMELS-H Forecasting Preprocessing — memory-optimized numpy pipeline.

Drop-in replacement for `preprocess_camelsh_forecast.py`. Same npz output
schema, dtype, basin/feature order, and (within float32 precision) values.
Replaces xarray Dataset construction and rolling operations with direct
numpy operations to avoid the ~50 GB memory peak of the xarray version.

Semantic guarantees preserved:
  * Same npz keys, shapes, dtypes
  * Same basin order (alphabetical, via pd.Index sort_values)
  * Same time order (chronological)
  * Same StandardScaler-equivalent X scaling (any-feature-NaN row dropped)
  * Same per-basin Y scaling
  * Same cumulative features (Rainf_sum / Tair_avg / PotEvap_sum at 24/72/168h)
  * Same distance matrix algorithm

Differences vs the xarray version:
  * Rolling/cumulative features computed via float64 cumsum+diff cast to
    float32 (instead of xarray.rolling). Numerically equivalent at ~1e-7.
"""

import gc
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist


# ======================== Configuration ========================

DYNAMIC_VARS = [
    "CAPE", "CRainf_frac", "LWdown", "PotEvap", "PSurf",
    "Qair", "Rainf", "SWdown", "Tair", "Wind_E", "Wind_N",
]

STATIC_VARS = [
    "p_mean", "pet_mean", "aridity", "p_seasonality", "frac_snow",
    "high_prec_freq", "high_prec_dur", "low_prec_freq", "low_prec_dur",
    "ele_mt_sav", "slp_dg_uav", "ria_ha_usu", "run_mm_syr", "gwt_cm_sav",
    "cly_pc_uav", "slt_pc_uav", "snd_pc_uav", "kar_pc_use", "prm_pc_use",
    "pac_pc_use", "crp_pc_use", "for_pc_use", "urb_pc_use", "area_sqkm",
]

TARGET_VAR = "Q_camelsh_obs_norm"

CUMUL_WINDOWS = [24, 72, 168]
CUMUL_DEFS = [
    ("Rainf",   "sum"),
    ("Tair",    "avg"),
    ("PotEvap", "sum"),
]

VAL_DATES   = ("1995-01-01 00:00:00", "1996-12-31 23:00:00")
TRAIN_DATES = ("1997-01-01 00:00:00", "2018-12-31 23:00:00")
TEST_DATES  = ("2019-01-01 00:00:00", "2022-12-31 23:00:00")


# ======================== Loading ========================

def load_parquet_streaming(parquet_path, dynamic_vars, static_vars, target_var,
                            select_basins=None):
    """
    Per-basin streaming load → returns numpy dicts (no xarray).

    Returns
    -------
    dynamic     : dict[var -> (n_time, n_basin) float32]
    target      : (n_time, n_basin) float32
    static      : dict[var -> (n_basin,) float32]   (incl. latitude, longitude)
    times       : (n_time,) datetime64[ns]
    basin_names : (n_basin,) np.array of object (basin id strings)
    """
    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.num_row_groups

    all_static = list(static_vars)
    for v in ("latitude", "longitude"):
        if v not in all_static:
            all_static.append(v)

    # ------------------------------------------------------------------
    # IMPORTANT: row groups are NOT one-per-basin. This parquet is a flat
    # table sorted by (basin_id, Time) cut into fixed-size row groups, so a
    # single row group spans several basins AND a basin can straddle row-group
    # boundaries. Group rows by basin_id (NOT by row group) and align each
    # basin's values onto a canonical time axis by timestamp. (The old code
    # assumed 1 row group == 1 basin and silently scrambled/dropped basins.)
    # ------------------------------------------------------------------

    # Pass 1: stream basin_id of every row group -> ordered unique basin list
    seen, seen_set = [], set()
    for rg_idx in range(n_row_groups):
        bids = pf.read_row_group(rg_idx, columns=["basin_id"]).column("basin_id").to_numpy(zero_copy_only=False)
        for b in pd.unique(bids):
            bs = str(b)
            if bs not in seen_set:
                seen_set.add(bs)
                seen.append(bs)
    unique_basins = sorted(seen)
    if select_basins is not None:
        sel = set(str(b) for b in select_basins)
        unique_basins = [b for b in unique_basins if b in sel]
    n_basins = len(unique_basins)
    bid2idx = {b: i for i, b in enumerate(unique_basins)}
    print(f"Found {n_basins} basins (grouped by basin_id across {n_row_groups} row groups)")

    # Pass 2: canonical time axis = timestamps of ONE basin (first basin of RG0)
    t0 = pf.read_row_group(0, columns=["basin_id", "Time"])
    b0 = t0.column("basin_id").to_numpy(zero_copy_only=False)
    times0 = pd.to_datetime(t0.column("Time").to_numpy(zero_copy_only=False)).values
    times = np.unique(times0[b0 == b0[0]])
    n_times = len(times)
    print(f"Canonical time axis: {n_times} steps "
          f"[{pd.Timestamp(times[0])} .. {pd.Timestamp(times[-1])}] (from basin {str(b0[0])})")

    dynamic = {v: np.full((n_times, n_basins), np.nan, dtype=np.float32)
               for v in dynamic_vars}
    target = np.full((n_times, n_basins), np.nan, dtype=np.float32)
    static = {v: np.full(n_basins, np.nan, dtype=np.float32) for v in all_static}
    static_set = np.zeros(n_basins, dtype=bool)

    wanted_cols = ["basin_id", "Time"] + dynamic_vars + [target_var] + all_static
    wanted_cols = list(dict.fromkeys(wanted_cols))

    # Pass 3: fill arrays, mapping EACH basin's rows (wherever they appear) by timestamp
    for rg_idx in range(n_row_groups):
        t = pf.read_row_group(rg_idx, columns=wanted_cols)
        bids = t.column("basin_id").to_numpy(zero_copy_only=False)
        rg_times = pd.to_datetime(t.column("Time").to_numpy(zero_copy_only=False)).values
        dyn_cols = {v: t.column(v).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                    for v in dynamic_vars}
        tgt_col = t.column(target_var).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        stat_cols = {v: t.column(v).to_numpy(zero_copy_only=False) for v in all_static}

        for b in pd.unique(bids):
            bs = str(b)
            if bs not in bid2idx:
                continue
            idx = bid2idx[bs]
            m = (bids == b)
            rt = rg_times[m]
            ti = np.searchsorted(times, rt)
            ti_clip = np.clip(ti, 0, n_times - 1)
            exact = times[ti_clip] == rt   # keep only exact timestamp matches
            ti_ok = ti_clip[exact]
            for var in dynamic_vars:
                dynamic[var][ti_ok, idx] = dyn_cols[var][m][exact]
            target[ti_ok, idx] = tgt_col[m][exact]
            if not static_set[idx]:
                for var in all_static:
                    raw = stat_cols[var][m][0]
                    try:
                        static[var][idx] = float(raw)
                    except (TypeError, ValueError):
                        static[var][idx] = np.nan
                static_set[idx] = True

        if (rg_idx + 1) % 20 == 0 or rg_idx + 1 == n_row_groups:
            print(f"  loaded {rg_idx + 1}/{n_row_groups} row groups")

    # Match the xarray version: let numpy auto-pick unicode dtype (e.g. <U7),
    # NOT object dtype (npz storage differs).
    basin_names = np.array(unique_basins)
    return dynamic, target, static, times, basin_names


# ======================== Cumulative features ========================

def rolling_sum_min1(arr, window):
    """
    Rolling sum with min_periods=1 along axis 0.
    float64 accumulator → cast back to float32.
    Equivalent to xarray .rolling(window, min_periods=1).sum() for NaN-free input.
    """
    csum = np.cumsum(arr, axis=0, dtype=np.float64)
    out = np.empty_like(csum)
    out[:window] = csum[:window]
    out[window:] = csum[window:] - csum[:-window]
    return out.astype(np.float32, copy=False)


def rolling_mean_min1(arr, window):
    """Rolling mean with min_periods=1 along axis 0."""
    rsum = rolling_sum_min1(arr, window).astype(np.float64, copy=False)
    counts = np.minimum(np.arange(1, len(arr) + 1), window).astype(np.float64)
    return (rsum / counts[:, None]).astype(np.float32, copy=False)


def build_cumulative_features(dynamic):
    """
    Returns dict {name: (n_time, n_basin) float32}.
    Naming and order MUST match the xarray version's x_vars extension:
        Rainf_sum_24h, Rainf_sum_72h, Rainf_sum_168h,
        Tair_avg_24h,  Tair_avg_72h,  Tair_avg_168h,
        PotEvap_sum_24h, PotEvap_sum_72h, PotEvap_sum_168h
    """
    cumul = {}
    for src_var, op in CUMUL_DEFS:
        suffix = "sum" if op == "sum" else "avg"
        for window in CUMUL_WINDOWS:
            name = f"{src_var}_{suffix}_{window}h"
            if op == "sum":
                cumul[name] = rolling_sum_min1(dynamic[src_var], window)
            else:
                cumul[name] = rolling_mean_min1(dynamic[src_var], window)
    return cumul


def cumul_feature_names():
    return [
        f"{src}_{'sum' if op == 'sum' else 'avg'}_{w}h"
        for (src, op) in CUMUL_DEFS for w in CUMUL_WINDOWS
    ]


# ======================== Scaling stats ========================

def _get_feature_2d(var, dynamic, static, cumul, n_times):
    """
    Return (n_time, n_basin) float32 view (or broadcast) for one X feature.
    Used by scaling-stat accumulation; broadcasted static is read-only.
    """
    if var in dynamic:
        return dynamic[var]
    if var in cumul:
        return cumul[var]
    if var in static:
        s = static[var]
        return np.broadcast_to(s[np.newaxis, :], (n_times, s.shape[0]))
    raise KeyError(f"Unknown feature: {var}")


def build_valid_mask(x_vars, dynamic, static, cumul, n_times, n_basins):
    """
    (n_time, n_basin) bool: True where ALL 44 x features are non-NaN.
    Matches xarray version's `~np.isnan(x_train_flat).any(axis=1)` filter.
    """
    mask = np.ones((n_times, n_basins), dtype=bool)
    for var in x_vars:
        col = _get_feature_2d(var, dynamic, static, cumul, n_times)
        mask &= ~np.isnan(col)
    return mask


def compute_x_scaling(x_vars, train_idx, valid_mask, dynamic, static, cumul,
                       n_times, n_basins):
    """
    Per-feature sum / sumsq over (train ∧ valid_mask), then mean/std.
    ddof=0 to match sklearn StandardScaler.
    """
    sub_mask = valid_mask[train_idx, :]
    total_count = int(sub_mask.sum())
    if total_count == 0:
        raise RuntimeError("No valid (non-NaN) train samples for X scaling.")

    n_feat = len(x_vars)
    sum_per   = np.zeros(n_feat, dtype=np.float64)
    sumsq_per = np.zeros(n_feat, dtype=np.float64)

    for f_idx, var in enumerate(x_vars):
        col = _get_feature_2d(var, dynamic, static, cumul, n_times)
        sub = col[train_idx, :][sub_mask].astype(np.float64, copy=False)
        sum_per[f_idx]   = sub.sum()
        sumsq_per[f_idx] = (sub ** 2).sum()

    mean = sum_per / total_count
    var  = sumsq_per / total_count - mean ** 2
    std  = np.sqrt(np.maximum(var, 0.0))
    # sklearn StandardScaler keeps tiny-variance features at scale_=1
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)  # keep float64 for save


def compute_y_scaling_per_basin(target, train_idx, n_basins):
    """Per-basin mean/std on non-NaN train values. Matches the original loop."""
    y_means = np.zeros(n_basins, dtype=np.float32)
    y_stds  = np.zeros(n_basins, dtype=np.float32)
    train_target = target[train_idx, :]
    for i in range(n_basins):
        bd = train_target[:, i]
        valid = bd[~np.isnan(bd)]
        if valid.size > 0:
            y_means[i] = float(np.mean(valid))
            s = float(np.std(valid))
            y_stds[i] = s if s >= 1e-8 else 1.0
        else:
            y_means[i] = 0.0
            y_stds[i] = 1.0
    return y_means, y_stds


# ======================== Build per-split arrays ========================

def build_split_arrays(split_idx, dynamic, target, static, cumul, x_vars,
                       x_mean, x_std, y_means, y_stds, n_times, n_basins):
    """
    Returns x_arr (n_basin, n_split_time, n_feat) and y_obs / y_raw (n_basin, t, 1).
    """
    n_split_time = int(split_idx.sum())
    n_feat = len(x_vars)
    eps = np.float32(1e-10)

    print(f"  building x_arr ({n_basins}, {n_split_time}, {n_feat}) "
          f"~ {n_basins * n_split_time * n_feat * 4 / 1e9:.2f} GB")
    x_arr = np.empty((n_basins, n_split_time, n_feat), dtype=np.float32)

    for f_idx, var in enumerate(x_vars):
        mu = np.float32(x_mean[f_idx])
        sigma = np.float32(x_std[f_idx]) + eps
        col = _get_feature_2d(var, dynamic, static, cumul, n_times)
        # col[split_idx, :] is (n_split_time, n_basins); transpose to (n_basins, n_split_time)
        x_arr[:, :, f_idx] = (col[split_idx, :].T - mu) / sigma

    y_obs = np.empty((n_basins, n_split_time, 1), dtype=np.float32)
    y_raw = np.empty((n_basins, n_split_time, 1), dtype=np.float32)
    eps_y = np.float32(1e-10)
    for i in range(n_basins):
        y_basin = target[split_idx, i]
        y_raw[i, :, 0] = y_basin
        mu_i = y_means[i]
        sigma_i = y_stds[i] + eps_y
        y_obs[i, :, 0] = (y_basin - mu_i) / sigma_i

    return x_arr, y_obs, y_raw


# ======================== Distance matrix ========================

def create_distance_matrix(static, n_basins):
    """Same algorithm as the xarray version (float64 coords, euclidean + sigmoid)."""
    coords = np.zeros((n_basins, 2))
    coords[:, 0] = static["latitude"].astype(np.float64)
    coords[:, 1] = static["longitude"].astype(np.float64)
    dist_raw = cdist(coords, coords, metric="euclidean")
    adj = -dist_raw
    nz = adj != 0
    mean_adj = float(np.mean(adj[nz]))
    std_adj  = float(np.std(adj[nz]))
    adj[nz] = (adj[nz] - mean_adj) / std_adj
    adj[nz] = 1.0 / (1.0 + np.exp(-adj[nz]))
    return adj, mean_adj, std_adj


# ======================== Main pipeline ========================

def prep_data(parquet_file, out_file, select_basins=None):
    print("Loading CAMELS-H hourly data...")
    dynamic, target, static, times, basin_names = load_parquet_streaming(
        parquet_file, DYNAMIC_VARS, STATIC_VARS, TARGET_VAR,
        select_basins=select_basins,
    )
    n_times = len(times)
    n_basins = len(basin_names)

    print("Adding cumulative features...")
    cumul = build_cumulative_features(dynamic)

    x_vars = list(DYNAMIC_VARS) + list(STATIC_VARS) + cumul_feature_names()

    print("Splitting data into train/val/test...")
    times_pd = pd.to_datetime(times)
    train_idx = np.asarray(
        (times_pd >= pd.Timestamp(TRAIN_DATES[0])) &
        (times_pd <= pd.Timestamp(TRAIN_DATES[1])))
    val_idx = np.asarray(
        (times_pd >= pd.Timestamp(VAL_DATES[0])) &
        (times_pd <= pd.Timestamp(VAL_DATES[1])))
    test_idx = np.asarray(
        (times_pd >= pd.Timestamp(TEST_DATES[0])) &
        (times_pd <= pd.Timestamp(TEST_DATES[1])))

    print("Scaling features...")
    valid_mask = build_valid_mask(x_vars, dynamic, static, cumul, n_times, n_basins)
    x_mean, x_std = compute_x_scaling(x_vars, train_idx, valid_mask,
                                       dynamic, static, cumul, n_times, n_basins)

    print("Calculating per-basin Y scaling parameters...")
    y_means, y_stds = compute_y_scaling_per_basin(target, train_idx, n_basins)

    print("Applying per-basin Y scaling...")
    print("Converting to numpy arrays...")
    x_trn, y_obs_trn, y_raw_trn = build_split_arrays(
        train_idx, dynamic, target, static, cumul, x_vars,
        x_mean, x_std, y_means, y_stds, n_times, n_basins)
    x_val, y_obs_val, y_raw_val = build_split_arrays(
        val_idx, dynamic, target, static, cumul, x_vars,
        x_mean, x_std, y_means, y_stds, n_times, n_basins)
    x_tst, y_obs_tst, y_raw_tst = build_split_arrays(
        test_idx, dynamic, target, static, cumul, x_vars,
        x_mean, x_std, y_means, y_stds, n_times, n_basins)

    print("Creating distance matrix...")
    dist_matrix, dist_mean, dist_std = create_distance_matrix(static, n_basins)

    times_trn = times[train_idx]
    times_val = times[val_idx]
    times_tst = times[test_idx]

    # Free large intermediates before npz save: dynamic / cumul / target / static
    # are no longer needed once x_*/y_* and dist_matrix are built.
    del dynamic, cumul, target, static
    gc.collect()

    data_dict = {
        "x_trn": x_trn, "y_obs_trn": y_obs_trn, "y_raw_trn": y_raw_trn,
        "x_val": x_val, "y_obs_val": y_obs_val, "y_raw_val": y_raw_val,
        "x_tst": x_tst, "y_obs_tst": y_obs_tst, "y_raw_tst": y_raw_tst,
        "times_trn": times_trn, "times_val": times_val, "times_tst": times_tst,
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_means, "y_std": y_stds,
        "basin_names": basin_names,
        "n_segs": n_basins,
        "dist_matrix": dist_matrix,
        "dist_mean": dist_mean, "dist_std": dist_std,
        "x_vars": np.array(x_vars),
        "y_obs_vars": np.array([TARGET_VAR]),
    }

    print(f"Saving preprocessed data to {out_file}...")
    # Uncompressed savez avoids the ~5 GB compression buffer (~10 GB more on disk).
    np.savez(out_file, **data_dict)

    print("\nPreprocessing complete!")
    print(f"  Basins:            {n_basins}")
    print(f"  Dynamic features:  {len(DYNAMIC_VARS)}")
    print(f"  Static features:   {len(STATIC_VARS)}")
    print(f"  Total x features:  {len(x_vars)}")
    print(f"  Target:            {TARGET_VAR}")
    print(f"  Train times:       {x_trn.shape[1]}")
    print(f"  Val   times:       {x_val.shape[1]}")
    print(f"  Test  times:       {x_tst.shape[1]}")
    print(f"  x_trn shape:       {x_trn.shape}")
    print(f"  y_obs_trn shape:   {y_obs_trn.shape}")

    return data_dict


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # CLI args (matches the xarray version): [basin1 basin2 ...]
    # Optional --out PATH to control output path (default data/prepped.npz).
    out_file = os.path.join(data_dir, "prepped.npz")
    args = list(sys.argv[1:])
    if "--out" in args:
        i = args.index("--out")
        out_file = args[i + 1]
        args = args[:i] + args[i + 2:]
    select_basins = args if args else None
    if select_basins:
        print(f"Selected basins: {select_basins}")

    prep_data(
        parquet_file="../../camelsh_global.parquet",
        out_file=out_file,
        select_basins=select_basins,
    )
