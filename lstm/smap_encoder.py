"""
SMAP pixel encoder + LSTM fusion model.

Each basin-day is a variable-length set of in-basin 9 km pixels
(2 values each: sm_surface, sm_rootzone). Pixels are kept in a flat ragged
layout (no padding): all valid pixels of the batch are concatenated along
one axis, and seg_id maps each pixel to its sample in the batch.

    token = Linear(sm 2-dim) + MLP(pixel offset 2-dim)
    -> shared per-pixel MLP (DeepSets)
    -> readout by seg_id  => one embedding per basin-day

Two readouts, selected by SMAP_READOUT (baked into the model at build time
and carried by the whole-model checkpoint):
  mean  equal-weight scatter mean over the pixel set (original scheme)
  attn  single-head cross-attention: a query attends over the basin's own
        pixels, so the weighting is content-dependent instead of uniform.
        SMAP_QUERY_CTX (comma list) chooses what the query is built from:
          grid_size     pixel count + bounding-grid width/height
          pixel_mean    mean of the basin's pixel tokens (data-derived context)
          static_attrs  static_enc basin attributes
        All context projections are zero-initialized: at the start of
        training the query is the learned base vector alone.

LSTMWithSMAP concatenates that embedding onto the forcing features X and
feeds the existing pure-LSTM backbone (lstm/model.py), trained end-to-end
with the same masked-RMSE objective. No contrastive alignment: the task
loss aligns the modalities, and sm carries information complementary to
forcing that an alignment objective would suppress.
"""

import torch
import torch.nn as nn

from model import LSTM


import os


class SMAPEncoder(nn.Module):
    """DeepSets over each sample's pixel set, ragged (padding-free) layout.

    Optional attribute-conditioned encoding (SMAP_ENC_ATTRS=1): each pixel
    token additionally receives a projection of the basin's physical
    attributes, so the representation of the moisture signal depends on the
    terrain that produced it. The attribute projection is zero-initialized:
    at the start of training the encoder is identical to the unconditioned
    one."""

    def __init__(self, d_model=32, dropout=0.1, n_attrs=16):
        super().__init__()
        self.readout = os.environ.get('SMAP_READOUT', 'mean')
        ctx = os.environ.get('SMAP_QUERY_CTX', '')
        self.query_ctx = tuple(c for c in ctx.split(',') if c)
        unknown = set(self.query_ctx) - {'grid_size', 'pixel_mean', 'static_attrs'}
        if self.readout not in ('mean', 'attn') or unknown:
            raise ValueError(f"SMAP_READOUT={self.readout}, unknown ctx {unknown}")
        # token-side attr conditioning belongs to the mean branch only; with
        # attn the attrs enter through the query instead. use_attrs is what
        # consumers check: does this encoder consume basin attrs at all?
        self.attrs_in_tokens = os.environ.get('SMAP_ENC_ATTRS') == '1' and self.readout == 'mean'
        self.use_attrs = self.attrs_in_tokens or 'static_attrs' in self.query_ctx
        if self.attrs_in_tokens:
            self.attr_proj = nn.Sequential(
                nn.Linear(n_attrs, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            nn.init.zeros_(self.attr_proj[-1].weight)
            nn.init.zeros_(self.attr_proj[-1].bias)
        if self.readout == 'attn':
            self.q_base = nn.Parameter(torch.randn(d_model) * d_model ** -0.5)
            self.q_norm = nn.LayerNorm(d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            ctx_in_dims = {'grid_size': 3, 'pixel_mean': d_model, 'static_attrs': n_attrs}
            self.ctx_proj = nn.ModuleDict()
            for name in self.query_ctx:
                proj = nn.Linear(ctx_in_dims[name], d_model)
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                self.ctx_proj[name] = proj
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

    def forward(self, sm, xy, seg_id, n_samples, attrs=None):
        """
        sm        : (N, T, 2) float — normalized values of every valid pixel
                    in the batch (N = total pixels across samples)
        xy        : (N, 2)    float — pixel offsets (grid units)
        seg_id    : (N,)      long  — sample index of each pixel
        n_samples : int             — batch size B
        attrs     : (B, K)    float — per-basin static attributes (optional)
        Returns   : (B, T, d_model) — one embedding per sample-day
        """
        N, T, _ = sm.shape
        tokens = self.val_proj(sm) + self.pos_proj(xy)[:, None, :]      # (N,T,d)
        if self.attrs_in_tokens and attrs is not None:
            tokens = tokens + self.attr_proj(attrs)[seg_id][:, None, :]
        tokens = self.pixel_mlp(tokens)

        counts = torch.bincount(seg_id, minlength=n_samples).clamp(min=1)
        if self.readout == 'mean':
            pooled = self._segment_mean(tokens, seg_id, n_samples, counts)
        else:
            pooled = self._attn_readout(tokens, xy, seg_id, n_samples, counts, attrs)
        return self.out_proj(pooled)                                    # (B,T,d)

    def _segment_mean(self, tokens, seg_id, n_samples, counts):
        pooled = tokens.new_zeros(n_samples, tokens.shape[1], self.d_model)
        pooled.index_add_(0, seg_id, tokens)                            # sum per sample
        return pooled / counts[:, None, None].float()

    def _build_query(self, tokens, xy, seg_id, n_samples, counts, attrs):
        """Sum of the learned base vector and the selected context projections
        (all zero-initialized), normalized. Shape (B,T,d) after broadcast."""
        q = self.q_base.view(1, 1, -1)
        if 'grid_size' in self.query_ctx:
            lo = xy.new_full((n_samples, 2), float('inf'))
            hi = xy.new_full((n_samples, 2), float('-inf'))
            lo.index_reduce_(0, seg_id, xy, 'amin', include_self=True)
            hi.index_reduce_(0, seg_id, xy, 'amax', include_self=True)
            grid_size = torch.log1p(torch.cat([counts[:, None].float(), hi - lo + 1], dim=1))
            q = q + self.ctx_proj['grid_size'](grid_size)[:, None, :]           # (B,1,d)
        if 'pixel_mean' in self.query_ctx:
            pixel_mean = self._segment_mean(tokens, seg_id, n_samples, counts)
            q = q + self.ctx_proj['pixel_mean'](pixel_mean)                     # (B,T,d)
        if 'static_attrs' in self.query_ctx and attrs is not None:
            q = q + self.ctx_proj['static_attrs'](attrs)[:, None, :]            # (B,1,d)
        return self.q_norm(q).expand(n_samples, tokens.shape[1], self.d_model)

    def _attn_readout(self, tokens, xy, seg_id, n_samples, counts, attrs):
        """Single-head cross-attention: the query attends over its own
        sample's pixel tokens; softmax runs within each segment (ragged,
        no padding)."""
        q = self._build_query(tokens, xy, seg_id, n_samples, counts, attrs)
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        scores = (q[seg_id] * k).sum(-1) * self.d_model ** -0.5         # (N,T)
        seg_max = scores.new_full((n_samples, scores.shape[1]), float('-inf'))
        seg_max.index_reduce_(0, seg_id, scores, 'amax', include_self=True)
        w = torch.exp(scores - seg_max[seg_id])
        seg_sum = w.new_zeros(n_samples, w.shape[1])
        seg_sum.index_add_(0, seg_id, w)
        w = w / seg_sum[seg_id].clamp(min=1e-12)                        # softmax per segment
        out = tokens.new_zeros(n_samples, tokens.shape[1], self.d_model)
        out.index_add_(0, seg_id, w.unsqueeze(-1) * v)
        return out                                                      # (B,T,d)


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

    def forward(self, x, sm, xy, seg_id, attrs=None):
        """
        x  : (B, T, F) — forcing + static features (from prepped.npz)
        sm/xy/seg_id/attrs : see SMAPEncoder (ragged pixel layout)
        """
        emb = self.smap_encoder(sm, xy, seg_id, n_samples=x.shape[0], attrs=attrs)
        return self.lstm(torch.cat([x, emb], dim=-1))


if __name__ == '__main__':
    # smoke test: 4 basins with different pixel counts, ragged layout
    B, T, F = 4, 365, 48
    n_px = [5, 18, 40, 2]
    N = sum(n_px)
    sm = torch.randn(N, T, 2)
    xy = torch.randn(N, 2)
    seg = torch.repeat_interleave(torch.arange(B), torch.tensor(n_px))
    attrs = torch.randn(B, 16)

    for readout, ctx in [('mean', ''), ('attn', ''),
                         ('attn', 'grid_size'),
                         ('attn', 'grid_size,pixel_mean'),
                         ('attn', 'grid_size,pixel_mean,static_attrs')]:
        os.environ['SMAP_READOUT'] = readout
        os.environ['SMAP_QUERY_CTX'] = ctx
        model = LSTMWithSMAP(input_dim=F, hidden_dim=20, d_smap=32)
        y = model(torch.randn(B, T, F), sm, xy, seg,
                  attrs=attrs if model.smap_encoder.use_attrs else None)
        assert y.shape == (B, T, 1), y.shape
        y.sum().backward()
        grads = sum(p.grad.abs().sum().item()
                    for p in model.smap_encoder.parameters() if p.grad is not None)
        print(f"readout={readout:4s} ctx=[{ctx}] output {tuple(y.shape)}, "
              f"encoder grad {grads:.3f}")
