"""
Transformer backbone for FM + Diffusion sequence modeling.

DiT-style design (Peebles & Xie 2023, "Scalable Diffusion Models with
Transformers") adapted for our (covariates, perturbed_target, t) input.

Architecture
------------
  1.  Concat x_cov + y_t                    -> (B, L, fused_dim)
  2.  Patch embed: Conv1d(fused_dim, d_model, k=3, pad=1)
                                            -> (B, L, d_model)
  3.  Time embedding (sinusoidal -> MLP)    -> (B, time_dim)
  4.  N x AdaLN-zero TransformerBlock:
        x = x + gate1 * Attn(AdaLN(x, t))
        x = x + gate2 * FFN(AdaLN(x, t))
      All `gate`, `scale`, `shift` are produced by a per-block MLP from the
      time embedding. The final layer of that MLP is zero-initialised so
      every block starts as identity. This is the critical training-
      stability trick from DiT.
  5.  LayerNorm -> Linear(d_model, target_dim)
                                            -> (B, L, target_dim)

Why this design (vs DLinear)
----------------------------
  * Self-attention along the TIME axis directly expresses cross-time +
    cross-channel relations like "rainfall at t-12 + temperature at t-6
    -> flow rise at t". DLinear can only represent per-channel temporal
    mixing + per-time channel projection, never both at once.
  * AdaLN-zero injection lets each block adapt its computation per
    diffusion-time t (different processing at noisy t -> data t), which
    is essential because the score / velocity field changes shape with
    t. Plain FiLM at the input is too coarse.
  * Zero-init residual gates + zero-init output projection -> network
    starts as a no-op. With diffusion losses this stops the early-
    training catastrophe where score blows up at t->0.

API parity
----------
The constructor accepts (and silently ignores) `kernel_size` and
`individual` so it is drop-in interchangeable with `DLinearBackboneTS`
in coupled_fmdiff.py:

    self.score_net = TransformerBackboneTS(
        cov_dim=cov_dim, target_dim=1, seq_len=seq_len,
        d_model=d_model, n_heads=n_heads, n_blocks=n_blocks,
        time_embed_dim=time_embed_dim, dropout=dropout,
    )

The forward signature is identical: forward(x_cov, y_t, t).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embedding (same as DLinearBackboneTS for consistency)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Multi-head self-attention along time axis
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must divide n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head)
        # (3, B, H, L, d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        # Use PyTorch's fused SDPA when available (FlashAttention etc.)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0
        )
        # (B, H, L, d_head) -> (B, L, D)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj_drop(self.proj(out))


# ---------------------------------------------------------------------------
# Position-wise feedforward
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(self, d_model, mult=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult)
        self.fc2 = nn.Linear(d_model * mult, d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ---------------------------------------------------------------------------
# DiT-style block with AdaLN-zero conditioning
# ---------------------------------------------------------------------------
class DiTBlock(nn.Module):
    """
    x = x + gate1(t) * Attn(AdaLN1(x, t))
    x = x + gate2(t) * FFN(AdaLN2(x, t))

    A single per-block MLP produces 6 vectors:
        (scale1, shift1, gate1, scale2, shift2, gate2)
    The final layer of that MLP is zero-initialised, so at t=0 the
    block is exactly the identity (gates are zero, scales are zero
    so AdaLN reduces to plain LayerNorm with bias 0).
    """

    def __init__(self, d_model, n_heads, time_dim,
                 dropout=0.0, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = FFN(d_model, mult=ffn_mult, dropout=dropout)

        # 6 * d_model: scale1, shift1, gate1, scale2, shift2, gate2
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 6 * d_model),
        )
        # zero-init final layer -> block starts as identity
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x, t_emb):
        # x: (B, L, D); t_emb: (B, time_dim)
        m = self.modulation(t_emb)               # (B, 6*D)
        s1, b1, g1, s2, b2, g2 = m.chunk(6, dim=-1)
        # broadcast over time axis (B, 1, D)
        s1, b1, g1 = s1.unsqueeze(1), b1.unsqueeze(1), g1.unsqueeze(1)
        s2, b2, g2 = s2.unsqueeze(1), b2.unsqueeze(1), g2.unsqueeze(1)

        # AdaLN1 + attention + gated residual
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h)

        # AdaLN2 + FFN + gated residual
        h = self.norm2(x) * (1 + s2) + b2
        x = x + g2 * self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# Top-level backbone
# ---------------------------------------------------------------------------
class TransformerBackboneTS(nn.Module):
    """
    Drop-in replacement for DLinearBackboneTS, accepting the same forward
    signature `(x_cov, y_t, t) -> output`.

    Args:
        cov_dim:         covariate channels
        target_dim:      target channels (typically 1)
        seq_len:         sequence length (only used for shape printing /
                         checks; attention is content-based not position-
                         indexed)
        d_model:         hidden width (default 128)
        n_heads:         attention heads (default 8)
        n_blocks:        number of DiT blocks (default 4)
        time_embed_dim:  sinusoidal time embedding dim AND modulation MLP
                         hidden dim (default 128)
        dropout:         attention + FFN dropout (default 0.1)
        ffn_mult:        FFN expansion factor (default 4)
        kernel_size:     IGNORED — accepted for DLinear API parity
        individual:      IGNORED — accepted for DLinear API parity
    """

    def __init__(self, cov_dim, target_dim, seq_len,
                 d_model=128, n_heads=8, n_blocks=4,
                 time_embed_dim=128, dropout=0.1, ffn_mult=4,
                 kernel_size=None, individual=None):
        super().__init__()
        self.cov_dim = cov_dim
        self.target_dim = target_dim
        self.seq_len = seq_len
        self.d_model = d_model

        fused_dim = cov_dim + target_dim

        # Patch / token embedding: Conv1d on time axis (k=3, padding=1)
        self.patch_embed = nn.Conv1d(
            fused_dim, d_model, kernel_size=3, padding=1
        )

        # Time embedding -> MLP -> per-block conditioning
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Stack of DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, time_embed_dim,
                     dropout=dropout, ffn_mult=ffn_mult)
            for _ in range(n_blocks)
        ])

        # Output head
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, target_dim)

        self._init_weights()

    def _init_weights(self):
        # Patch embed: standard conv init
        nn.init.kaiming_normal_(self.patch_embed.weight,
                                 mode='fan_out', nonlinearity='linear')
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        # Time MLP
        for m in self.time_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Inside-block modulations are already zero-init in DiTBlock.
        # Output projection zero-init: at start the network outputs 0,
        # which is the correct prior mean for normalised data.
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x_cov, y_t, t):
        """
        x_cov: (B, L, cov_dim)
        y_t:   (B, L, target_dim)
        t:     (B,) in [0, 1]
        Returns: (B, L, target_dim)
        """
        fused = torch.cat([x_cov, y_t], dim=-1)             # (B, L, F+1)

        # Patch embed: Conv1d expects (B, C, L)
        h = self.patch_embed(fused.permute(0, 2, 1))         # (B, D, L)
        h = h.permute(0, 2, 1)                                # (B, L, D)

        # Time conditioning
        t_emb = self.time_mlp(self.time_embed(t))            # (B, time_dim)

        # Transformer blocks
        for blk in self.blocks:
            h = blk(h, t_emb)

        # Output
        h = self.norm_out(h)
        out = self.proj_out(h)                                # (B, L, target_dim)
        return out
