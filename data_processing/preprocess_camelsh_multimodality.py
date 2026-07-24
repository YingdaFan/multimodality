"""
CAMELSH Daily Data Preprocessing (multimodality pipeline) — pure NumPy.

Daily variant of preprocess_perseg_aligntime_camelsh.py, aligned to the SMAP
overlap window (2015-04-01 onward) for the multimodal experiments. Reads
camelsh_global_daily.parquet (618 basins, daily, 57 cols) produced by
CAMELSH/make_camelsh_daily.py.

Differences vs the hourly variant:
  * Parquet: camelsh_global_daily.parquet (daily rows, same schema)
  * Time window: 2015-04-01 .. 2024-12-31 (SMAP era only)
  * Cumulative windows in DAYS: 7/14/30 (names *_7d etc.)
  * Temporal features: doy_sin/doy_cos only (no hour terms)
  * seq_len in days (default 365, offset 1.0 — CAMELS daily convention)

Same npz output schema as the other perseg preprocess variants, so the
LSTM stage-1 / diffusion stage-2 pipelines consume it unchanged. The SMAP
grids themselves are NOT packed here — they ride in a side-channel npz
keyed by (basin_id, date) built separately.
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist


# ---------------------------------------------------------------------------
# Variable lists
# ---------------------------------------------------------------------------

# 11 NLDAS-2 forcing variables (input X, time-varying)
DYNAMIC_VARS = [
    'Tair', 'Qair', 'PSurf', 'Wind_E', 'Wind_N',
    'LWdown', 'SWdown', 'CRainf_frac', 'CAPE', 'PotEvap', 'Rainf',
]
# 28 static attributes loaded from parquet. latitude/longitude are loaded
# for distance-matrix computation only; the remaining 26 enter X.
LOAD_STATIC_VARS = [
    'latitude', 'longitude',
    'area_sqkm', 'elev_mean', 'elev_max', 'elev_min', 'slope_pct',
    'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow',
    'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
    'frac_forest', 'frac_developed',
    'sand_frac', 'silt_frac', 'clay_frac',
    'soil_permeability', 'soil_awc', 'rock_depth',
    'baseflow_index', 'runoff_mean',
    't_avg_basin', 'precip_avg_basin',
    # modality-trust modulators: karst bypasses SMAP-visible storage,
    # regulation/lakes decouple Q from rainfall-runoff dynamics
    'kar_pc_use', 'dor_pc_pva', 'lka_pc_use',
]
# 26 static features going into X (skip lat/lng — these are for dist matrix)
X_STATIC_VARS = [v for v in LOAD_STATIC_VARS if v not in ('latitude', 'longitude')]

TARGET_VAR = 'Q_camelsh_obs_norm'

CUMUL_WINDOWS = [7, 14, 30]  # in days
CUMUL_DEFS = [
    ('Rainf',   'sum'),
    ('Tair',    'avg'),
    ('PotEvap', 'sum'),
]

TEMP_VARS = ['doy_sin', 'doy_cos']


# ---------------------------------------------------------------------------
# Per-basin streaming load (numpy dict — no xarray)
# ---------------------------------------------------------------------------

def load_camelsh_arrays(parquet_path, select_basins=None):
    """
    Stream-load the parquet one row group at a time.
    Returns plain numpy arrays.

    Returns
    -------
    times       : (n_time,) datetime64[ns]
    basin_names : (n_basin,) <U... unicode array of basin IDs
    dynamic     : dict[var -> (n_time, n_basin) float32]   for forcing X
    target      : (n_time, n_basin) float32                 for Y
    static      : dict[var -> (n_basin,) float32]           for static
    """
    pf = pq.ParquetFile(parquet_path)
    n_row_groups = pf.num_row_groups
    print(f'Parquet: {n_row_groups} row groups')

    # ------------------------------------------------------------------
    # Row groups are NOT one-per-basin: flat table sorted by
    # (basin_id, Time) cut into fixed-size row groups. Group rows by
    # basin_id and align onto a canonical time axis by timestamp.
    # ------------------------------------------------------------------

    # Pass 1: stream basin_id of every row group -> ordered unique basin list
    seen, seen_set = [], set()
    for rg_idx in range(n_row_groups):
        bids = pf.read_row_group(rg_idx, columns=['basin_id']).column('basin_id').to_numpy(zero_copy_only=False)
        for b in pd.unique(bids):
            bs = str(b)
            if bs not in seen_set:
                seen_set.add(bs)
                seen.append(bs)
    unique_basins = sorted(seen)
    if select_basins is not None:
        select_set = set(str(b) for b in select_basins)
        unique_basins = [b for b in unique_basins if b in select_set]
        if not unique_basins:
            raise ValueError(f'No basins matched select_basins={select_basins}')
    n_basins = len(unique_basins)
    bid2idx = {b: i for i, b in enumerate(unique_basins)}
    print(f'Found {n_basins} basins (grouped by basin_id across {n_row_groups} row groups)')

    # Pass 2: canonical time axis = timestamps of the first basin in row
    # group 0 (contiguous, time-sorted). All basins share this daily grid.
    t0 = pf.read_row_group(0, columns=['basin_id', 'Time'])
    b0 = t0.column('basin_id').to_numpy(zero_copy_only=False)
    times0 = pd.to_datetime(t0.column('Time').to_numpy(zero_copy_only=False)).values
    times = np.unique(times0[b0 == b0[0]])
    n_times = len(times)
    print(f'Canonical time axis: {n_times} steps '
          f'[{pd.Timestamp(times[0])} .. {pd.Timestamp(times[-1])}] '
          f'(from basin {str(b0[0])})')

    # Preallocate output arrays
    dynamic = {v: np.full((n_times, n_basins), np.nan, dtype=np.float32)
               for v in DYNAMIC_VARS}
    target = np.full((n_times, n_basins), np.nan, dtype=np.float32)
    static = {v: np.full(n_basins, np.nan, dtype=np.float32)
              for v in LOAD_STATIC_VARS}
    static_set = np.zeros(n_basins, dtype=bool)

    # Pass 3: fill arrays, mapping each basin's rows by timestamp
    wanted = ['basin_id', 'Time'] + DYNAMIC_VARS + [TARGET_VAR] + LOAD_STATIC_VARS
    for rg_idx in range(n_row_groups):
        t = pf.read_row_group(rg_idx, columns=wanted)
        bids = t.column('basin_id').to_numpy(zero_copy_only=False)
        rg_times = pd.to_datetime(t.column('Time').to_numpy(zero_copy_only=False)).values
        dyn_cols = {v: t.column(v).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                    for v in DYNAMIC_VARS}
        tgt_col = t.column(TARGET_VAR).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        stat_cols = {v: t.column(v).to_numpy(zero_copy_only=False) for v in LOAD_STATIC_VARS}

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
            for v in DYNAMIC_VARS:
                dynamic[v][ti_ok, idx] = dyn_cols[v][m][exact]
            target[ti_ok, idx] = tgt_col[m][exact]
            if not static_set[idx]:
                for v in LOAD_STATIC_VARS:
                    raw = stat_cols[v][m][0]
                    try:
                        static[v][idx] = float(raw)
                    except (TypeError, ValueError):
                        static[v][idx] = np.nan
                static_set[idx] = True
        if (rg_idx + 1) % 20 == 0:
            print(f'  loaded {rg_idx + 1}/{n_row_groups} row groups')

    return times, np.array(unique_basins), dynamic, target, static


# ---------------------------------------------------------------------------
# Feature engineering — numpy
# ---------------------------------------------------------------------------

def add_temporal_features_np(times):
    """Return (n_time, 2) float32 of [doy_sin, doy_cos]."""
    times_pd = pd.to_datetime(times)
    doy = times_pd.dayofyear.values.astype(np.float32)
    out = np.empty((len(times), len(TEMP_VARS)), dtype=np.float32)
    out[:, 0] = np.sin(2 * np.pi * doy / 365)
    out[:, 1] = np.cos(2 * np.pi * doy / 365)
    return out


def rolling_sum_min1(arr, window):
    """
    Rolling sum with min_periods=1 along axis 0.
    arr: (n_time, n_basin) float32 (forcing X has no NaN)
    """
    csum = np.cumsum(arr, axis=0, dtype=np.float64)
    out = np.empty_like(csum)
    out[:window] = csum[:window]
    out[window:] = csum[window:] - csum[:-window]
    return out.astype(np.float32, copy=False)


def rolling_mean_min1(arr, window):
    """Rolling mean with min_periods=1."""
    rsum = rolling_sum_min1(arr, window).astype(np.float64, copy=False)
    counts = np.minimum(np.arange(1, len(arr) + 1), window).astype(np.float64)
    return (rsum / counts[:, None]).astype(np.float32, copy=False)


def build_cumulative_features(dynamic):
    """
    Build all cumulative features. Naming: '{src}_{op}_{window}d'
    e.g. 'Rainf_sum_7d', 'Tair_avg_14d'.
    """
    cumul = {}
    for window in CUMUL_WINDOWS:
        for src_var, op in CUMUL_DEFS:
            name = f'{src_var}_{"sum" if op == "sum" else "avg"}_{window}d'
            if op == 'sum':
                cumul[name] = rolling_sum_min1(dynamic[src_var], window)
            else:
                cumul[name] = rolling_mean_min1(dynamic[src_var], window)
    return cumul


# ---------------------------------------------------------------------------
# X scaling stats — streaming per-feature (no flat materialise)
# ---------------------------------------------------------------------------

def build_valid_mask(x_vars, train_idx, dynamic, static, cumul, temp_feats,
                      n_basin):
    """
    (n_train_time, n_basin) bool: True iff every X feature is non-NaN at
    that (time, basin).
    """
    n_train_time = int(train_idx.sum())
    mask = np.ones((n_train_time, n_basin), dtype=bool)
    for var in x_vars:
        if var in DYNAMIC_VARS:
            mask &= ~np.isnan(dynamic[var][train_idx, :])
        elif var in X_STATIC_VARS:
            mask &= ~np.isnan(static[var])[None, :]
        elif var in cumul:
            mask &= ~np.isnan(cumul[var][train_idx, :])
        elif var in TEMP_VARS:
            ti = TEMP_VARS.index(var)
            mask &= ~np.isnan(temp_feats[train_idx, ti])[:, None]
        else:
            raise KeyError(f'Unknown feature: {var}')
    return mask


def compute_x_scaling_streaming(x_vars, train_idx,
                                  dynamic, static, cumul, temp_feats, n_basin):
    """
    Per-feature mean & std over (train ∧ valid_mask). Uses sum / sum_sq
    accumulators (ddof=0; matches sklearn StandardScaler).
    """
    valid_mask = build_valid_mask(x_vars, train_idx, dynamic, static, cumul,
                                   temp_feats, n_basin)
    total_count = int(valid_mask.sum())
    if total_count == 0:
        raise RuntimeError('No valid (non-NaN) train samples for X scaling.')

    count_per_basin = valid_mask.sum(axis=0).astype(np.float64)  # (n_basin,)
    count_per_time  = valid_mask.sum(axis=1).astype(np.float64)  # (n_train_time,)

    n_feat = len(x_vars)
    sum_per   = np.zeros(n_feat, dtype=np.float64)
    sumsq_per = np.zeros(n_feat, dtype=np.float64)

    for f_idx, var in enumerate(x_vars):
        if var in DYNAMIC_VARS:
            col = dynamic[var][train_idx, :][valid_mask].astype(np.float64, copy=False)
            sum_per[f_idx]   = col.sum()
            sumsq_per[f_idx] = (col ** 2).sum()
        elif var in X_STATIC_VARS:
            s = static[var].astype(np.float64)
            sum_per[f_idx]   = (s * count_per_basin).sum()
            sumsq_per[f_idx] = (s * s * count_per_basin).sum()
        elif var in cumul:
            col = cumul[var][train_idx, :][valid_mask].astype(np.float64, copy=False)
            sum_per[f_idx]   = col.sum()
            sumsq_per[f_idx] = (col ** 2).sum()
        elif var in TEMP_VARS:
            ti = TEMP_VARS.index(var)
            t_col = temp_feats[train_idx, ti].astype(np.float64)
            sum_per[f_idx]   = (t_col * count_per_time).sum()
            sumsq_per[f_idx] = (t_col * t_col * count_per_time).sum()
        else:
            raise KeyError(f'Unknown feature: {var}')

    mean = (sum_per / total_count).astype(np.float32)
    var  = (sumsq_per / total_count) - (sum_per / total_count) ** 2
    std  = np.sqrt(np.maximum(var, 0)).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


# ---------------------------------------------------------------------------
# Per-basin Y scaling (NaN-aware, train slice only)
# ---------------------------------------------------------------------------

def compute_y_scaling_per_basin(target, train_idx, n_basin):
    y_means = np.zeros(n_basin, dtype=np.float32)
    y_stds  = np.zeros(n_basin, dtype=np.float32)
    train_target = target[train_idx, :]
    for i in range(n_basin):
        y = train_target[:, i]
        valid = ~np.isnan(y)
        if valid.any():
            v = y[valid]
            y_means[i] = float(v.mean(dtype=np.float64))
            s = float(v.std(dtype=np.float64))
            y_stds[i] = s if s >= 1e-8 else 1.0
        else:
            y_means[i] = np.nan
            y_stds[i]  = np.nan
    return y_means, y_stds


# ---------------------------------------------------------------------------
# Build per-split (n_basin, n_split_time, n_feat) X and (n_basin, t, 1) Y
# ---------------------------------------------------------------------------

def build_split_arrays(split_idx, dynamic, target, static, cumul, temp_feats,
                       x_vars, x_mean, x_std, y_means, y_stds, n_basin):
    """
    Returns x_arr (n_basin, n_split_time, n_feat), y_obs_arr, y_raw_arr.
    """
    n_split_time = int(split_idx.sum())
    n_feat = len(x_vars)
    eps = np.float32(1e-10)

    print(f'  building x_arr shape ({n_basin}, {n_split_time}, {n_feat}) '
          f'~ {n_basin * n_split_time * n_feat * 4 / 1e9:.1f} GB')
    x_arr = np.empty((n_basin, n_split_time, n_feat), dtype=np.float32)

    for f_idx, var in enumerate(x_vars):
        mu = x_mean[f_idx]
        sigma = x_std[f_idx] + eps
        if var in DYNAMIC_VARS:
            x_arr[:, :, f_idx] = (dynamic[var][split_idx, :].T - mu) / sigma
        elif var in X_STATIC_VARS:
            x_arr[:, :, f_idx] = (static[var][:, None] - mu) / sigma
        elif var in cumul:
            x_arr[:, :, f_idx] = (cumul[var][split_idx, :].T - mu) / sigma
        elif var in TEMP_VARS:
            ti = TEMP_VARS.index(var)
            x_arr[:, :, f_idx] = (temp_feats[split_idx, ti][None, :] - mu) / sigma
        else:
            raise KeyError(f'Unknown feature: {var}')

    y_obs_arr = np.empty((n_basin, n_split_time, 1), dtype=np.float32)
    y_raw_arr = np.empty((n_basin, n_split_time, 1), dtype=np.float32)
    for i in range(n_basin):
        y_basin = target[split_idx, i]
        y_raw_arr[i, :, 0] = y_basin
        mu_i = y_means[i]
        sigma_i = y_stds[i] + eps
        y_obs_arr[i, :, 0] = (y_basin - mu_i) / sigma_i

    return x_arr, y_obs_arr, y_raw_arr


# ---------------------------------------------------------------------------
# Window batching — numpy
# ---------------------------------------------------------------------------

def convert_batch_reshape_np(arr, seq_len, offset):
    """
    Sliding-window batching.
    arr: (n_basin, n_time, n_feat)  float32
    Returns: (n_windows × n_basin, seq_len, n_feat) same dtype.
    """
    n_basin, n_time, n_feat = arr.shape
    if offset > 1:
        step = int(offset)
    else:
        step = int(seq_len * offset)
    n_windows = (n_time - seq_len) // step + 1

    out = np.empty((n_windows * n_basin, seq_len, n_feat), dtype=arr.dtype)
    sample_idx = 0
    for w in range(n_windows):
        s = w * step
        e = s + seq_len
        for b in range(n_basin):
            out[sample_idx] = arr[b, s:e, :]
            sample_idx += 1
    return out


def create_ids_times_arrays_np(basin_names, dates, n_windows, seq_len, offset):
    n_segs = len(basin_names)
    n_samples = n_windows * n_segs
    ids = np.empty((n_samples, seq_len), dtype=object)
    times_arr = np.empty((n_samples, seq_len), dtype='datetime64[ns]')
    if offset > 1:
        step = int(offset)
    else:
        step = int(seq_len * offset)
    sample_idx = 0
    for w in range(n_windows):
        s = w * step
        e = s + seq_len
        window_dates = dates[s:e]
        for b in range(n_segs):
            ids[sample_idx, :] = basin_names[b]
            times_arr[sample_idx, :] = window_dates
            sample_idx += 1
    return ids[..., np.newaxis], times_arr[..., np.newaxis]


# ---------------------------------------------------------------------------
# Distance matrix (same algorithm as the other preprocess variants)
# ---------------------------------------------------------------------------

def create_distance_matrix_np(static, basin_names):
    n = len(basin_names)
    coords = np.zeros((n, 2), dtype=np.float64)
    coords[:, 0] = static['latitude'].astype(np.float64)
    coords[:, 1] = static['longitude'].astype(np.float64)
    dist_raw = cdist(coords, coords, metric='euclidean')
    adj = -dist_raw
    nz = adj != 0
    if nz.any():
        m = float(np.mean(adj[nz]))
        s = float(np.std(adj[nz]))
        adj[nz] = (adj[nz] - m) / s
        adj[nz] = 1.0 / (1.0 + np.exp(-adj[nz]))
    else:
        m, s = 0.0, 1.0
    A_hat = adj + np.eye(n)
    D = A_hat.sum(axis=1)
    A_hat = np.diag(D ** -1.0) @ A_hat
    return A_hat, m, s


# ---------------------------------------------------------------------------
# Main entry: prep_data
# ---------------------------------------------------------------------------

def prep_data(parquet_file, train_dates, val_dates, test_dates,
              seq_len=365, offset=1.0, out_file='prepped.npz',
              add_temporal=True, add_cumulative=True, select_basins=None):
    """
    Full preprocessing pipeline. Same npz output schema as the other
    perseg preprocess variants.
    """
    print('Loading CAMELSH daily data...')
    times, basin_names, dynamic, target, static = load_camelsh_arrays(
        parquet_file, select_basins=select_basins
    )
    n_time, n_basin = target.shape
    print(f'Loaded: {n_basin} basins x {n_time} timesteps')

    # ---- Feature engineering ----
    cumul = {}
    if add_cumulative:
        print('Adding cumulative features (cumsum trick)...')
        cumul = build_cumulative_features(dynamic)

    temp_feats = None
    if add_temporal:
        print('Adding temporal features...')
        temp_feats = add_temporal_features_np(times)

    # ---- Build x_vars list ----
    x_vars = list(DYNAMIC_VARS) + list(X_STATIC_VARS)
    if add_temporal:
        x_vars += TEMP_VARS
    if add_cumulative:
        for window in CUMUL_WINDOWS:
            for src_var, op in CUMUL_DEFS:
                x_vars.append(f'{src_var}_{"sum" if op == "sum" else "avg"}_{window}d')
    n_feat = len(x_vars)
    print(f'Total features: {n_feat}')

    # ---- Time index for each split ----
    train_start = pd.Timestamp(train_dates[0]); train_end = pd.Timestamp(train_dates[1])
    val_start   = pd.Timestamp(val_dates[0]);   val_end   = pd.Timestamp(val_dates[1])
    tst_start   = pd.Timestamp(test_dates[0]);  tst_end   = pd.Timestamp(test_dates[1])

    train_idx = (times >= train_start) & (times <= train_end)
    val_idx   = (times >= val_start)   & (times <= val_end)
    tst_idx   = (times >= tst_start)   & (times <= tst_end)

    # ---- Scaling stats (from train slice only) ----
    print('Computing X scaling stats (streaming per-feature)...')
    x_mean, x_std = compute_x_scaling_streaming(
        x_vars, train_idx, dynamic, static, cumul, temp_feats, n_basin
    )

    print('Computing per-basin Y scaling stats...')
    y_means, y_stds = compute_y_scaling_per_basin(target, train_idx, n_basin)

    # ---- Per-split processing ----
    splits = [
        ('trn', train_idx, train_start, train_end, offset),
        ('val', val_idx,   val_start,   val_end,   1.0),
        ('tst', tst_idx,   tst_start,   tst_end,   1.0),
    ]

    data_dict = {}
    for split_name, split_idx, s_start, s_end, split_offset in splits:
        print(f'\nProcessing {split_name} split {s_start.date()} -> {s_end.date()}...')
        split_times = times[split_idx]
        x_arr, y_obs_arr, y_raw_arr = build_split_arrays(
            split_idx, dynamic, target, static, cumul, temp_feats,
            x_vars, x_mean, x_std, y_means, y_stds, n_basin,
        )
        n_split_time = x_arr.shape[1]

        print(f'  window batching (seq_len={seq_len}, offset={split_offset})...')
        x_batched     = convert_batch_reshape_np(x_arr, seq_len, split_offset)
        y_obs_batched = convert_batch_reshape_np(y_obs_arr, seq_len, split_offset)
        y_raw_batched = convert_batch_reshape_np(y_raw_arr, seq_len, split_offset)

        if split_offset > 1:
            step = int(split_offset)
        else:
            step = int(seq_len * split_offset)
        n_windows = (n_split_time - seq_len) // step + 1

        ids_arr, times_arr_split = create_ids_times_arrays_np(
            basin_names, split_times, n_windows, seq_len, split_offset
        )

        del x_arr, y_obs_arr, y_raw_arr

        data_dict[f'x_{split_name}']     = x_batched
        data_dict[f'y_obs_{split_name}'] = y_obs_batched
        data_dict[f'y_raw_{split_name}'] = y_raw_batched
        data_dict[f'ids_{split_name}']   = ids_arr
        data_dict[f'times_{split_name}'] = times_arr_split
        print(f'  {split_name} done: x_shape={x_batched.shape}')

    # ---- Distance matrix ----
    print('\nComputing distance matrix...')
    dist_matrix, dist_mean, dist_std = create_distance_matrix_np(static, basin_names)

    # ---- Metadata (same keys as the other variants) ----
    data_dict.update({
        'dist_matrix': dist_matrix,
        'x_vars':      np.array(x_vars),
        'y_obs_vars':  np.array([TARGET_VAR]),
        'x_mean':      x_mean,
        'x_std':       x_std,
        'y_mean':      y_means,
        'y_std':       y_stds,
        'dist_mean':   np.float32(dist_mean),
        'dist_std':    np.float32(dist_std),
        'n_segs':      n_basin,
        'basin_names': basin_names,
    })

    print(f'\nSaving to {out_file} (compressed)...')
    np.savez_compressed(out_file, **data_dict)

    print('\nPreprocessing complete!')
    print(f'Number of basins:    {n_basin}')
    print(f'Training samples:    {data_dict["x_trn"].shape[0]}')
    print(f'Validation samples:  {data_dict["x_val"].shape[0]}')
    print(f'Test samples:        {data_dict["x_tst"].shape[0]}')
    print(f'Features:            {n_feat}')
    print(f'Sequence length:     {seq_len}')
    return data_dict


# ---------------------------------------------------------------------------
# CLI: same usage as the other preprocess variants
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # SMAP overlap window (SMAP starts 2015-04-01, parquet ends 2024-12-31).
    # imputation 覆盖整段, val 是 train 内部时间 held-out (用于 early stopping)
    train_dates = ('2015-04-01', '2024-12-31')
    val_dates   = ('2022-01-01', '2023-12-31')
    test_dates  = ('2015-04-01', '2024-12-31')

    basins = None
    if len(sys.argv) > 1:
        basins = sys.argv[1:]
        print(f'Selected basins: {basins}')

    seq_len = 365       # 365 days = 1 year (CAMELS daily convention)

    prep_data(
        parquet_file='../../camelsh_global_daily.parquet',
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        seq_len=seq_len,
        offset=1.0,
        out_file=os.path.join(data_dir, 'prepped.npz'),
        add_temporal=True,
        add_cumulative=True,
        select_basins=basins,
    )
