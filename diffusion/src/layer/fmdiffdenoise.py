"""
FMDiff Denoiser — variant of denoise.py for the Stochastic Interpolant
calibration pipeline (sical_gx_enc.py).

Identical to denoise.py except ConditionalGuidedModel.forward accepts an
`apply_sigma_softplus` flag. With apply_sigma_softplus=False, the second
output is the raw linear projection (allowed to be negative), which lets
SI repurpose this head as the denoiser eta(t, x) = E[z | x_t = x].

The original denoise.py is left untouched so diffcal/fmcal continue to use
the softplus-constrained sigma head as before.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out


class XConditionNetwork(nn.Module):
    """X feature condition network with cross-attention. Same as denoise.py."""

    def __init__(self, d_model, y_state_dim, x_cond_dim=64, n_heads=4, dropout=0.1):
        super(XConditionNetwork, self).__init__()
        self.d_model = d_model
        self.x_cond_dim = x_cond_dim

        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, x_cond_dim, kernel_size=3, padding=1),
        )
        self.local_norm = nn.LayerNorm(x_cond_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=x_cond_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(x_cond_dim)

        self.y_proj = nn.Linear(y_state_dim, x_cond_dim)

        self.output_proj = nn.Sequential(
            nn.Linear(x_cond_dim, x_cond_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(x_cond_dim, x_cond_dim),
        )
        self.output_norm = nn.LayerNorm(x_cond_dim)

    def forward(self, x_enc, y_state):
        B, T, _ = x_enc.shape
        O = y_state.shape[1]

        x_local = self.local_conv(x_enc.transpose(1, 2)).transpose(1, 2)
        x_local = self.local_norm(x_local)

        if T < O:
            x_local = F.interpolate(
                x_local.transpose(1, 2),
                size=O,
                mode='linear',
                align_corners=True,
            ).transpose(1, 2)

        y_query = self.y_proj(y_state)

        attn_out, _ = self.cross_attn(
            query=y_query,
            key=x_local,
            value=x_local,
        )
        attn_out = self.attn_norm(attn_out + y_query)

        x_cond = self.output_proj(attn_out)
        x_cond = self.output_norm(x_cond + attn_out)

        return x_cond


class ConditionalGuidedModel(nn.Module):
    """Conditional guided denoiser with optional softplus toggle on second head.

    First head (out_proj):
        - In SI:    velocity b(t, x)
        - In NsDiff: noise prediction eps_pred
    Second head (sigma_lin):
        - apply_sigma_softplus=True  -> softplus(raw), positive variance (NsDiff style)
        - apply_sigma_softplus=False -> raw linear output, can be negative
                                        (SI's denoiser eta = E[z | x_t])
    """

    def __init__(self, diff_steps, enc_in, c_out=None, d_model=32,
                 x_cond_dim=64, n_heads=4, dropout=0.1):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = diff_steps + 1

        if c_out is None:
            c_out = enc_in
        self.c_out = c_out
        self.enc_in = enc_in
        self.d_model = d_model
        self.x_cond_dim = x_cond_dim

        self.x_condition_net = XConditionNetwork(
            d_model=d_model,
            y_state_dim=c_out * 2,
            x_cond_dim=x_cond_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        data_dim = c_out * 3 + x_cond_dim
        hidden_dim = 256

        self.lin1 = ConditionalLinear(data_dim, hidden_dim, n_steps)
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin4 = ConditionalLinear(hidden_dim, 128, n_steps)

        self.out_proj = nn.Linear(128, c_out)
        self.sigma_lin = nn.Linear(128, c_out)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y_t, y_0_hat, g_x, t, apply_sigma_softplus=True):
        """
        Args:
            x: encoder output (B, T, d_model)
            y_t: noisy/interpolated sample (B, O, c_out)
            y_0_hat: conditional prior (B, O, c_out)
            g_x: variance estimate slot (B, O, c_out); ones for SI
            t: discrete diffusion timestep tensor
            apply_sigma_softplus: if False, return raw second-head output (for SI eta)

        Returns:
            head1: (B, O, c_out) — eps_pred (NsDiff) or velocity b (SI)
            head2: (B, O, c_out) — sigma (positive, NsDiff) or eta (raw, SI)
        """
        y_state = torch.cat([y_t, y_0_hat], dim=-1)
        x_cond = self.x_condition_net(x, y_state)

        h = torch.cat((y_t, y_0_hat, g_x, x_cond), dim=-1)

        h1 = F.silu(self.lin1(h, t))
        h1 = self.norm1(h1)
        h1 = self.dropout(h1)

        h2 = F.silu(self.lin2(h1, t))
        h2 = self.norm2(h2)
        h2 = h2 + h1
        h2 = self.dropout(h2)

        h3 = F.silu(self.lin3(h2, t))
        h3 = self.norm3(h3)
        h3 = h3 + h2
        h3 = self.dropout(h3)

        h4 = F.silu(self.lin4(h3, t))

        head1 = self.out_proj(h4)
        head2_raw = self.sigma_lin(h4)
        head2 = F.softplus(head2_raw) if apply_sigma_softplus else head2_raw

        return head1, head2
