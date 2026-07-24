"""
DLinear backbone for diffusion / FM time-series modeling.

Adapted (verbatim) from flowmatchingdiffusion.zip. Self-contained.

Key design:
  - Accepts a noisy/perturbed target y_t in addition to covariates.
  - Accepts a continuous time embedding t in [0, 1].
  - Output shape matches y_t (it's a noise/score/velocity prediction,
    not a regression target).
  - Series decomposition (moving average) splits trend + seasonal.
  - FiLM-style time conditioning (scale + shift) on channel axis.
"""

import math
import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) -> (B, dim)
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class DLinearBackboneTS(nn.Module):
    """
    DLinear-style network that maps (covariates, perturbed target, t) -> output
    of the same shape as the perturbed target.

    Args:
        cov_dim:        number of covariate channels
        target_dim:     number of target channels (typically 1 for streamflow)
        seq_len:        sequence length
        kernel_size:    moving-average window for series decomp
        time_embed_dim: dim of sinusoidal time embedding
        individual:     if True, use per-channel linear maps (more params,
                        usually worse on small data)
        dropout:        dropout on the fused features
    """

    def __init__(
        self,
        cov_dim,
        target_dim,
        seq_len,
        kernel_size=25,
        time_embed_dim=64,
        individual=False,
        dropout=0.0,
    ):
        super().__init__()
        self.cov_dim = cov_dim
        self.target_dim = target_dim
        self.seq_len = seq_len
        self.individual = individual

        fused_dim = cov_dim + target_dim
        self.fused_dim = fused_dim

        self.decomp = SeriesDecomp(kernel_size)

        if individual:
            self.lin_seasonal = nn.ModuleList(
                [nn.Linear(seq_len, seq_len) for _ in range(fused_dim)]
            )
            self.lin_trend = nn.ModuleList(
                [nn.Linear(seq_len, seq_len) for _ in range(fused_dim)]
            )
        else:
            self.lin_seasonal = nn.Linear(seq_len, seq_len)
            self.lin_trend = nn.Linear(seq_len, seq_len)

        # FiLM-style time conditioning: per-channel (scale, shift)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_film = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, fused_dim * 2),  # scale + shift
        )

        self.output_proj = nn.Linear(fused_dim, target_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        if self.individual:
            for layer in list(self.lin_seasonal) + list(self.lin_trend):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        else:
            for layer in [self.lin_seasonal, self.lin_trend]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        # FiLM final layer initialised to identity-ish (scale~1, shift~0)
        last = self.time_film[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x_cov, y_t, t):
        """
        Args:
            x_cov: (B, L, cov_dim)
            y_t:   (B, L, target_dim)
            t:     (B,)
        Returns:
            (B, L, target_dim)
        """
        fused = torch.cat([x_cov, y_t], dim=-1)  # (B, L, fused_dim)

        # FiLM on the channel axis
        film = self.time_film(self.time_embed(t))         # (B, fused_dim*2)
        scale, shift = film.chunk(2, dim=-1)              # (B, fused_dim) each
        # last layer init to 0 -> (1+0) is identity at start
        fused = fused * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        seasonal, trend = self.decomp(fused)

        if self.individual:
            seas_out = torch.zeros_like(seasonal)
            tr_out = torch.zeros_like(trend)
            for i in range(self.fused_dim):
                seas_out[:, :, i] = self.lin_seasonal[i](seasonal[:, :, i].clone())
                tr_out[:, :, i] = self.lin_trend[i](trend[:, :, i].clone())
        else:
            seas_out = self.lin_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            tr_out = self.lin_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        out = seas_out + tr_out
        out = self.dropout(out)
        out = self.output_proj(out)  # (B, L, target_dim)
        return out
