"""
Pack SMAP nc cutouts into one npz for the multimodality pipeline.

For each basin in camelsh_global_daily.parquet, read its SMAP nc, keep only
in-basin pixels (watershed_mask == 1), and store:

  sm_<basin>  : (n_time, n_px, 2) float32  — [sm_surface, sm_rootzone]
  xy_<basin>  : (n_px, 2)         float32  — pixel offsets from basin
                centroid in units of the 9 km grid pitch (for positional
                encoding; scale is comparable across basins)

plus shared keys:

  times       : (n_time,) datetime64[ns]  — daily axis, aligned to the
                preprocess window (2015-04-01 .. 2024-12-31)
  basin_names : (n_basin,) str
  sm_mean, sm_std : (2,) float32 — global normalization stats over all
                in-basin pixels (train-window computed; the encoder applies
                (sm - mean) / std)

Usage:
    python pack_smap.py            # all basins in the parquet
    python pack_smap.py 01046500 ...  # subset
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import pyarrow.parquet as pq
import pyarrow.compute as pc

SMAP_DIR = '/home/yif47/river-dl/temporal/CAMELSH/moisture/CAMELSH'
PARQUET = '../../camelsh_global_daily.parquet'
DATE_START, DATE_END = '2015-04-01', '2024-12-31'
OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'smap_packed.npz')


def main():
    basins = [str(b).zfill(8) for b in pc.unique(
        pq.ParquetFile(PARQUET).read(columns=['basin_id'])['basin_id']).to_pylist()]
    if len(sys.argv) > 1:
        want = set(sys.argv[1:])
        basins = [b for b in basins if b in want]
    basins = sorted(basins)
    print(f'{len(basins)} basins')

    out = {}
    times = None
    # accumulators for global normalization stats
    tot_n, tot_sum, tot_sumsq = 0, np.zeros(2), np.zeros(2)

    for k, b in enumerate(basins):
        ds = xr.open_dataset(f'{SMAP_DIR}/SMAP_{b}.nc')
        t = pd.to_datetime(ds.time.values)
        sel = (t >= DATE_START) & (t <= DATE_END)
        if times is None:
            times = t[sel].values
        m = ds.watershed_mask.values == 1
        iy, ix = np.nonzero(m)
        sm = np.stack([ds.sm_surface.values[sel][:, iy, ix],
                       ds.sm_rootzone.values[sel][:, iy, ix]], axis=-1).astype(np.float32)
        xy = np.stack([iy - iy.mean(), ix - ix.mean()], axis=-1).astype(np.float32)
        ds.close()

        out[f'sm_{b}'] = sm
        out[f'xy_{b}'] = xy
        tot_n += sm.shape[0] * sm.shape[1]
        tot_sum += sm.sum(axis=(0, 1), dtype=np.float64)
        tot_sumsq += (sm.astype(np.float64) ** 2).sum(axis=(0, 1))
        if (k + 1) % 100 == 0:
            print(f'  {k + 1}/{len(basins)}')

    mean = tot_sum / tot_n
    std = np.sqrt(tot_sumsq / tot_n - mean ** 2)
    out['times'] = times
    out['basin_names'] = np.array(basins)
    out['sm_mean'] = mean.astype(np.float32)
    out['sm_std'] = std.astype(np.float32)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    print(f'Saving {OUT_FILE} ...')
    np.savez_compressed(OUT_FILE, **out)
    print(f'Done: {len(basins)} basins, {len(times)} days, '
          f'sm_mean={mean.round(4)}, sm_std={std.round(4)}')


if __name__ == '__main__':
    main()
