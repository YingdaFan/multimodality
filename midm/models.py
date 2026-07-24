"""
MIDM with Position-Invariant Denoiser for Spatial Extrapolation.

Keeps MIDM's diffusion framework (structured noise Σ, conditional sampling)
but replaces the EEM with a position-invariant denoiser:

    Per-basin shared encoder:
        [y_noisy_i, y_prev_c_i, obs_flag_i, (X_i)] → SharedEncoder → h_i
    Spatial interaction:
        [h_1,...,h_N] → SpatialAttention → [h'_1,...,h'_N]
    Per-basin shared decoder:
        h'_i → SharedDecoder → noise_pred_i

This is position-invariant: ALL basins use the SAME weights. Works for basins
never seen during training (spatial extrapolation), unlike the original EEM
which has fixed per-position weights via Linear(N → d_model).

Reference:
    Diffusion framework from Xu Wang et al., KDD 2023.
    Denoiser adapted for spatial extrapolation.
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t, dim):
    """Sinusoidal embedding for diffusion timestep."""
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    emb = t.float().unsqueeze(-1) * freq
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# 1. Learnable covariance  Sigma = diag(d^2) + U U^T
# ---------------------------------------------------------------------------

class LowRankCovariance(nn.Module):
    """
    Sigma = diag(d^2) + U U^T

    PSD by construction. Supports conditional sampling via Woodbury identity.
    """

    def __init__(self, n_vars, rank=8):
        super().__init__()
        self.n_vars = n_vars
        self.rank = rank
        self.log_d = nn.Parameter(torch.zeros(n_vars))
        self.U = nn.Parameter(torch.randn(n_vars, rank) * 0.01)

    def get_sigma(self):
        """Full covariance Sigma = diag(d^2) + U U^T. Shape: (N, N)."""
        d2 = self.log_d.exp() ** 2 + 1e-6
        return torch.diag(d2) + self.U @ self.U.T

    def sample_noise(self, shape, device):
        """Sample from N(0, Sigma). shape=(B, L), returns (B, L, N)."""
        B, L = shape
        N = self.n_vars
        Sigma = self.get_sigma()
        Sigma = Sigma + 1e-4 * torch.eye(N, device=device)
        try:
            Lchol = torch.linalg.cholesky(Sigma)
        except torch.linalg.LinAlgError:
            Sigma = Sigma + 1e-3 * torch.eye(N, device=device)
            Lchol = torch.linalg.cholesky(Sigma)
        z = torch.randn(B, L, N, device=device)
        return z @ Lchol.T

    def sample_conditional(self, obs_idx, miss_idx, x_obs_T):
        """
        Sample x_T^m ~ p(x_T^m | x_T^c) from conditional Gaussian.

        Args:
            obs_idx:  (n_obs,) int
            miss_idx: (n_miss,) int
            x_obs_T:  (B, L, n_obs)
        Returns:
            x_miss_T: (B, L, n_miss)
        """
        Sigma = self.get_sigma()
        device = Sigma.device

        Sigma_cc = Sigma[obs_idx][:, obs_idx]
        Sigma_mm = Sigma[miss_idx][:, miss_idx]
        Sigma_mc = Sigma[miss_idx][:, obs_idx]

        Sigma_cc_inv_Sigma_cm = torch.linalg.solve(Sigma_cc, Sigma_mc.T)

        B, L, n_obs = x_obs_T.shape
        x_flat = x_obs_T.reshape(B * L, n_obs)
        mu_flat = x_flat @ Sigma_cc_inv_Sigma_cm
        mu = mu_flat.reshape(B, L, -1)

        Sigma_cond = Sigma_mm - Sigma_mc @ Sigma_cc_inv_Sigma_cm
        Sigma_cond = Sigma_cond + 1e-4 * torch.eye(len(miss_idx), device=device)
        L_cond = torch.linalg.cholesky(Sigma_cond)

        n_miss = len(miss_idx)
        z = torch.randn(B, L, n_miss, device=mu.device)
        return mu + z @ L_cond.T


# ---------------------------------------------------------------------------
# 2. Diffusion schedule
# ---------------------------------------------------------------------------

class DiffusionSchedule:
    """Linear beta schedule."""

    def __init__(self, n_steps=50, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.n_steps = n_steps
        betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = self.alphas.cumprod(dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


# ---------------------------------------------------------------------------
# 3. Transformer block (shared by temporal and spatial attention)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block: self-attention + FFN."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 4. Position-Invariant Denoiser
# ---------------------------------------------------------------------------

class PositionInvariantDenoiser(nn.Module):
    """
    Position-invariant denoiser with alternating temporal/spatial attention.

    Every basin is processed by the SAME shared weights. Basin identity comes
    from its input features (Y values, X features, obs_flag), not from a
    fixed position in a weight matrix.

    Architecture per forward pass:
        1. Per-basin input: [y_noisy_i, y_prev_c_i, obs_flag_i, (X_i)]
        2. Shared input_proj → (B*N, L, d_model)
        3. Add diffusion step embedding + temporal position encoding
        4. Alternating layers:
           - Temporal attention: (B*N, L, d_model) — per-basin over time
           - Spatial attention:  (B*L, N, d_model) — per-timestep over basins
        5. Shared output_proj → noise prediction per basin
    """

    def __init__(self, input_dim, d_model=128, n_heads=4, n_layers=3,
                 max_seq_len=365, n_diffusion_steps=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Shared input projection: per-basin features → d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Diffusion timestep embedding
        self.t_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Temporal position encoding (learnable)
        self.pos_enc = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Alternating temporal and spatial attention layers
        self.temporal = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.spatial = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)])

        # Shared output
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, y_noisy, y_prev_c, obs_mask, t, x_features=None):
        """
        Args:
            y_noisy:    (B, N, L) — noisy Y at step t
            y_prev_c:   (B, N, L) — diffused observed Y at step t-1
            obs_mask:   (B, N, L) — 1=observed, 0=missing
            t:          (B,) int  — diffusion timestep
            x_features: (B, N, L, D_x) optional — exogenous features
        Returns:
            (B, N, L) — predicted noise
        """
        B, N, L = y_noisy.shape

        # Per-basin input: (B, N, L, input_dim)
        inputs = [
            y_noisy.unsqueeze(-1),    # (B, N, L, 1)
            y_prev_c.unsqueeze(-1),   # (B, N, L, 1)
            obs_mask.unsqueeze(-1),   # (B, N, L, 1)
        ]
        if x_features is not None:
            inputs.append(x_features)  # (B, N, L, D_x)
        per_basin = torch.cat(inputs, dim=-1)

        # Shared encoding: (B*N, L, input_dim) → (B*N, L, d_model)
        x = per_basin.reshape(B * N, L, -1)
        x = self.input_proj(x)

        # Add diffusion step embedding (broadcast over N and L)
        t_emb = sinusoidal_embedding(t, self.d_model).to(x.dtype)
        t_emb = self.t_proj(t_emb)                                   # (B, d_model)
        t_emb = t_emb[:, None, None, :].expand(-1, N, 1, -1)       # (B, N, 1, d_model)
        x = x + t_emb.reshape(B * N, 1, -1)

        # Add temporal position encoding (broadcast over B*N)
        x = x + self.pos_enc[:, :L, :]

        # Alternating temporal and spatial attention
        for t_blk, s_blk in zip(self.temporal, self.spatial):
            # Temporal: per-basin attention over time
            x = t_blk(x)                                            # (B*N, L, d_model)

            # Spatial: per-timestep attention over basins
            x = x.reshape(B, N, L, -1).permute(0, 2, 1, 3).reshape(B * L, N, -1)
            x = s_blk(x)                                            # (B*L, N, d_model)

            # Back to temporal layout
            x = x.reshape(B, L, N, -1).permute(0, 2, 1, 3).reshape(B * N, L, -1)

        # Output: per-basin noise prediction
        x = self.out_proj(self.out_norm(x))                         # (B*N, L, 1)
        return x.squeeze(-1).reshape(B, N, L)


# ---------------------------------------------------------------------------
# 5. MIDM model
# ---------------------------------------------------------------------------

class MIDM(nn.Module):
    """
    MIDM with position-invariant denoiser.

    Args:
        n_vars:    total number of basins
        d_x:       exogenous feature dim (0 = pure, >0 = with X conditioning)
        d_model:   hidden dim of denoiser
    """

    def __init__(self, n_vars, max_seq_len, d_model=128, n_heads=4,
                 n_layers=3, n_diffusion_steps=50, cov_rank=8,
                 d_x=0, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_x = d_x
        self.cov = LowRankCovariance(n_vars, rank=cov_rank)

        input_dim = 3 + d_x  # y_noisy, y_prev_c, obs_flag [+ x_features]
        self.denoiser = PositionInvariantDenoiser(
            input_dim=input_dim, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, max_seq_len=max_seq_len,
            n_diffusion_steps=n_diffusion_steps, dropout=dropout)

    def predict_noise(self, y_noisy, y_prev_c, obs_mask, t, x_features=None):
        # Auto-cast inputs to match denoiser dtype (BF16 if denoiser is BF16)
        dtype = next(self.denoiser.parameters()).dtype
        y_n = y_noisy.to(dtype)
        y_p = y_prev_c.to(dtype)
        om = obs_mask.to(dtype)
        xf = x_features.to(dtype) if x_features is not None else None
        return self.denoiser(y_n, y_p, om, t, xf).float()


# ---------------------------------------------------------------------------
# 6. DDIM Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_ddim(model, schedule, y_obs, obs_idx, miss_idx, n_steps=50,
                x_features=None):
    """
    DDIM sampling with MIDM conditional noise initialization.

    Args:
        model:      MIDM
        schedule:   DiffusionSchedule
        y_obs:      (B, n_obs, L) — clean observed Y (normalized)
        obs_idx:    (n_obs,) int
        miss_idx:   (n_miss,) int
        n_steps:    reverse steps
        x_features: (B, N, L, D_x) optional — exogenous features
    Returns:
        (B, n_miss, L) — imputed Y
    """
    model.eval()
    device = y_obs.device
    B, n_obs, L = y_obs.shape
    n_miss = len(miss_idx)
    N = model.n_vars
    total_T = schedule.n_steps

    # Build obs_mask: (B, N, L) — also handles NaN in observed basins
    obs_valid = (~torch.isnan(y_obs)).float()  # (B, n_obs, L)
    obs_mask = torch.zeros(B, N, L, device=device)
    obs_mask[:, obs_idx, :] = obs_valid

    # Clean NaN in y_obs
    y_obs_clean = y_obs.clone()
    y_obs_clean[torch.isnan(y_obs_clean)] = 0.0

    # --- Initialize via conditional noise sampling ---
    ab_T = schedule.alpha_bars[-1]
    noise_obs = model.cov.sample_noise((B, L), device)[:, :, obs_idx]
    x_obs_T = (torch.sqrt(ab_T) * y_obs_clean.permute(0, 2, 1)
               + torch.sqrt(1 - ab_T) * noise_obs)

    x_miss_T = model.cov.sample_conditional(obs_idx, miss_idx, x_obs_T)

    x_t = torch.zeros(B, N, L, device=device)
    x_t[:, obs_idx, :] = x_obs_T.permute(0, 2, 1)
    x_t[:, miss_idx, :] = x_miss_T.permute(0, 2, 1)

    # --- Reverse denoising ---
    if n_steps >= total_T:
        timesteps = list(range(total_T))[::-1]
    else:
        step_size = total_T / n_steps
        timesteps = [int((n_steps - 1 - i) * step_size) for i in range(n_steps)]

    for i, t_val in enumerate(timesteps):
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)
        ab_t = schedule.alpha_bars[t_val]

        # X_{t-1}^c: diffused observed at previous step
        if t_val > 0:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            ab_prev = schedule.alpha_bars[t_prev]
        else:
            ab_prev = torch.tensor(1.0, device=device)

        noise_for_obs = model.cov.sample_noise((B, L), device)[:, :, obs_idx]
        y_prev_c = torch.zeros(B, N, L, device=device)
        y_prev_c[:, obs_idx, :] = (
            torch.sqrt(ab_prev) * y_obs_clean
            + torch.sqrt(1 - ab_prev) * noise_for_obs.permute(0, 2, 1))

        # Predict noise
        eps = model.predict_noise(x_t, y_prev_c, obs_mask, t_batch, x_features)

        # Predict x_0
        x0_pred = (x_t - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        x0_pred = x0_pred.clamp(-50, 50)
        x0_pred[:, obs_idx, :] = y_obs_clean

        # Step to next timestep
        if i < len(timesteps) - 1:
            ab_next = schedule.alpha_bars[timesteps[i + 1]]
            x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps
        else:
            x_t = x0_pred

    return x_t[:, miss_idx, :]
