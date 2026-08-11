"""
SMAP batch provider for the multimodal LSTM.

Loads data_processing/data/smap_packed.npz (built by pack_smap.py) and, for
a batch of prepped.npz samples, returns the padded pixel tensors the
SMAPEncoder expects. Alignment is by (basin_id, window start date): each
prepped sample i covers times[i, 0..seq_len-1] for basin ids[i], and the
packed npz shares the same daily axis, so the slice is [t0 : t0+seq_len].
"""

import numpy as np
import pandas as pd
import torch


def normalize_sm(sm_raw, mean, std):
    """Single owner of the pixel-value normalization rule. Used by the
    training-time provider and by offline embedding computation alike."""
    return (sm_raw - mean) / std


@torch.no_grad()
def compute_basin_embeddings(encoder, packed_path, basins, attrs=None, device='cpu'):
    """Run a trained SMAPEncoder over each basin's full daily axis.

    encoder : SMAPEncoder in eval mode (weights already loaded)
    basins  : iterable of basin ids (str)
    attrs   : optional dict basin -> (K,) tensor, for attribute-conditioned
              encoders; None for the unconditioned encoder
    Returns : (emb, times) — emb: dict basin -> (n_days, d) float32 array
    """
    packed = np.load(packed_path, allow_pickle=True)
    mean, std = packed['sm_mean'], packed['sm_std']
    emb = {}
    for k, b in enumerate(sorted(basins)):
        sm = torch.from_numpy(normalize_sm(packed[f'sm_{b}'], mean, std))                   .transpose(0, 1).to(device)
        xy = torch.from_numpy(packed[f'xy_{b}']).to(device)
        seg = torch.zeros(sm.shape[0], dtype=torch.long, device=device)
        at = attrs[b][None].to(device) if attrs is not None else None
        emb[b] = encoder(sm, xy, seg, n_samples=1, attrs=at)[0].cpu().numpy()
        if (k + 1) % 100 == 0:
            print(f'  embeddings {k + 1}/{len(basins)}')
    return emb, packed['times']


class SMAPProvider:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        mean = d['sm_mean'].astype(np.float32)
        std = d['sm_std'].astype(np.float32)
        self.sm = {}
        self.xy = {}
        for b in d['basin_names']:
            b = str(b)
            self.sm[b] = torch.from_numpy(normalize_sm(d[f'sm_{b}'], mean, std))
            self.xy[b] = torch.from_numpy(d[f'xy_{b}'])
        self.times = pd.to_datetime(d['times']).values
        self.splits = {}
        self.attrs = None   # (basin -> (K,) tensor), set via set_static_attrs

    def set_static_attrs(self, basin_names, static_enc):
        """Register per-basin encoder attributes from prepped.npz."""
        self.attrs = {str(b): torch.from_numpy(np.asarray(static_enc[i], dtype='float32'))
                      for i, b in enumerate(basin_names)}

    def register_split(self, name, ids, times):
        """ids/times: (n_samples, seq_len, 1) arrays from prepped.npz."""
        basins = ids[:, 0, 0].astype(str)
        missing = set(basins) - set(self.sm)
        if missing:
            raise KeyError(f'basins not in smap_packed.npz: {sorted(missing)[:5]} ...')
        t0 = np.searchsorted(self.times, times[:, 0, 0].astype('datetime64[ns]'))
        if not (self.times[t0] == times[:, 0, 0]).all():
            raise ValueError(f'{name}: sample start dates not on the SMAP time axis')
        self.splits[name] = (basins, t0, times.shape[1])

    def batch(self, name, sample_indices, device):
        """
        Ragged (padding-free) batch for the given sample indices:
          sm     (N, seq_len, 2) — all valid pixels of the batch concatenated
          xy     (N, 2)
          seg_id (N,) long       — sample index (position in batch) per pixel
        """
        basins, t0, seq_len = self.splits[name]
        idx = sample_indices.tolist() if torch.is_tensor(sample_indices) else list(sample_indices)

        sm_parts, xy_parts, seg_parts = [], [], []
        for j, i in enumerate(idx):
            b = basins[i]
            # (seq_len, P, 2) -> (P, seq_len, 2): pixels on the flat axis
            sm_parts.append(self.sm[b][t0[i]:t0[i] + seq_len].transpose(0, 1))
            xy_parts.append(self.xy[b])
            seg_parts.append(torch.full((self.xy[b].shape[0],), j, dtype=torch.long))
        sm = torch.cat(sm_parts)
        xy = torch.cat(xy_parts)
        seg = torch.cat(seg_parts)
        out = [sm.to(device), xy.to(device), seg.to(device)]
        if self.attrs is not None:
            at = torch.stack([self.attrs[basins[i]] for i in idx])
            out.append(at.to(device))
        return tuple(out)
