"""
SMAP pixel encoder + LSTM fusion model.

Each basin-day is a variable-length set of in-basin 9 km pixels
(2 values each: sm_surface, sm_rootzone). Pixels are kept in a flat ragged
layout (no padding): all valid pixels of the batch are concatenated along
one axis, and seg_id maps each pixel to its sample in the batch.

    token = Linear(sm 2-dim) + MLP(pixel offset 2-dim)
    -> shared per-pixel MLP (DeepSets)
    -> scatter mean-pool by seg_id  => one embedding per basin-day

LSTMWithSMAP concatenates that embedding onto the forcing features X and
feeds the existing pure-LSTM backbone (lstm/model.py), trained end-to-end
with the same masked-RMSE objective. No contrastive alignment: the task
loss aligns the modalities, and sm carries information complementary to
forcing that an alignment objective would suppress.
"""

import torch
import torch.nn as nn

from model import LSTM


class SMAPEncoder(nn.Module):
    """DeepSets over each sample's pixel set, ragged (padding-free) layout."""

    def __init__(self, d_model=32, dropout=0.1):
        super().__init__()
        self.val_proj = nn.Linear(2, d_model)
        self.pos_proj = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.pixel_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
        )
        self.d_model = d_model

    def forward(self, sm, xy, seg_id, n_samples):
        """
        sm        : (N, T, 2) float — normalized values of every valid pixel
                    in the batch (N = total pixels across samples)
        xy        : (N, 2)    float — pixel offsets (grid units)
        seg_id    : (N,)      long  — sample index of each pixel
        n_samples : int             — batch size B
        Returns   : (B, T, d_model) — one embedding per sample-day
        """
        N, T, _ = sm.shape
        tokens = self.val_proj(sm) + self.pos_proj(xy)[:, None, :]      # (N,T,d)
        tokens = self.pixel_mlp(tokens)

        pooled = tokens.new_zeros(n_samples, T, self.d_model)
        pooled.index_add_(0, seg_id, tokens)                            # sum per sample
        counts = torch.bincount(seg_id, minlength=n_samples).clamp(min=1)
        pooled = pooled / counts[:, None, None].float()
        return self.out_proj(pooled)                                    # (B,T,d)


class LSTMWithSMAP(nn.Module):
    """Existing pure-LSTM backbone with the SMAP embedding concatenated to X."""

    def __init__(self, input_dim, hidden_dim, d_smap=32, adj_matrix=None,
                 recur_dropout=0, dropout=0, device='cpu', seed=None):
        super().__init__()
        self.input_dim = input_dim   # X width, without the embedding
        self.d_smap = d_smap
        self.smap_encoder = SMAPEncoder(d_model=d_smap)
        self.lstm = LSTM(input_dim=input_dim + d_smap, hidden_dim=hidden_dim,
                         adj_matrix=adj_matrix, recur_dropout=recur_dropout,
                         dropout=dropout, device=device, seed=seed)

    def forward(self, x, sm, xy, seg_id):
        """
        x  : (B, T, F) — forcing + static features (from prepped.npz)
        sm/xy/seg_id : see SMAPEncoder (ragged pixel layout)
        """
        emb = self.smap_encoder(sm, xy, seg_id, n_samples=x.shape[0])
        return self.lstm(torch.cat([x, emb], dim=-1))


if __name__ == '__main__':
    # smoke test: 4 basins with different pixel counts, ragged layout
    B, T, F = 4, 365, 48
    n_px = [5, 18, 40, 12]
    N = sum(n_px)
    sm = torch.randn(N, T, 2)
    xy = torch.randn(N, 2)
    seg = torch.repeat_interleave(torch.arange(B), torch.tensor(n_px))

    model = LSTMWithSMAP(input_dim=F, hidden_dim=20, d_smap=32)
    y = model(torch.randn(B, T, F), sm, xy, seg)
    assert y.shape == (B, T, 1), y.shape
    y.sum().backward()
    grads = sum(p.grad.abs().sum().item() for p in model.smap_encoder.parameters())
    print(f'output {tuple(y.shape)}, encoder grad magnitude {grads:.3f} (end-to-end OK)')
