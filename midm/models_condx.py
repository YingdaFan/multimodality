"""
MIDM with X conditioning (exogenous features) — per-basin injection.

Instead of cross-attention (which loses per-basin identity after input_proj),
this version concatenates per-basin X encodings directly to the Y values
BEFORE the EEM's input projection. This way input_proj sees:

    [Y_1, ..., Y_N, X̃_1_1, ..., X̃_1_k, ..., X̃_N_1, ..., X̃_N_k]

Each basin's Y value sits next to its own X encoding. The model can learn:
"basin #42 has Y=0 (missing) but X̃ indicates high elevation → reference
basins #15, #67 with similar X̃ and known Y."

No cross-attention needed — standard EEM self-attention suffices.

Reuses LowRankCovariance, DiffusionSchedule, EEMLayer from models.py.
"""

import torch
import torch.nn as nn

from models import LowRankCovariance, DiffusionSchedule, EEMLayer


# ---------------------------------------------------------------------------
# EEM with per-basin X injection
# ---------------------------------------------------------------------------

class EEMCondX(nn.Module):
    """
    Error Estimation Model with per-basin X feature injection.

    Input: Y values (N dims) + per-basin X encodings (N*d_x_enc dims)
           → total input dim = N*(1 + d_x_enc)

    Architecture is the same as pure EEM (transpose → embed → self-attention),
    just with a wider input that carries per-basin X identity.
    """

    def __init__(self, n_vars, max_seq_len, d_x, d_model=512, n_heads=8,
                 n_layers=4, n_diffusion_steps=50, d_x_enc=4, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_x_enc = d_x_enc
        self.d_model = d_model

        # Per-basin X encoder: D_x → d_x_enc per basin per timestep
        self.x_basin_proj = nn.Sequential(
            nn.Linear(d_x, d_x_enc * 4),
            nn.GELU(),
            nn.Linear(d_x_enc * 4, d_x_enc),
        )

        # Total input dim: N (Y values) + N * d_x_enc (X encodings)
        input_dim = n_vars * (1 + d_x_enc)

        # Step embedding (Sec 3.3.1): one learnable vector per diffusion step
        self.step_embed = nn.Embedding(n_diffusion_steps, input_dim)

        # Temporal embedding (Sec 3.3.2): sinusoidal positional encoding
        te = self._build_temporal_embedding(max_seq_len, input_dim)
        self.register_buffer('temporal_embed', te)

        # Input projection: enriched basin features → d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Standard EEM self-attention layers (no cross-attention needed)
        self.layers = nn.ModuleList([
            EEMLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection: d_model → N (only Y predictions, not X)
        self.output_proj = nn.Linear(d_model, n_vars)

    @staticmethod
    def _build_temporal_embedding(max_len, dim):
        """Sinusoidal temporal embedding (Eq 20)."""
        te = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000.0, torch.arange(0, dim, 2, dtype=torch.float) / dim)
        te[:, 0::2] = torch.sin(position / div_term)
        if dim % 2 == 1:
            te[:, 1::2] = torch.cos(position / div_term[:dim // 2])
        else:
            te[:, 1::2] = torch.cos(position / div_term)
        return te

    def forward(self, x_t, x_prev_c, t, x_features):
        """
        Args:
            x_t:        (B, N, L) — noisy Y at step t
            x_prev_c:   (B, N, L) — diffused observed Y at step t-1
            t:          (B,) int  — diffusion timestep
            x_features: (B, N, L, D_x) — exogenous features (ALL basins)
        Returns:
            (B, N, L) — predicted noise
        """
        B, N, L = x_t.shape

        # --- Y part: (B, N, L) → (B, L, N) ---
        y_input = (x_t + x_prev_c).permute(0, 2, 1)    # (B, L, N)

        # --- X part: encode per-basin per-timestep ---
        # (B, N, L, D_x) → (B, N, L, d_x_enc)
        x_enc = self.x_basin_proj(x_features)
        # → (B, L, N, d_x_enc) → (B, L, N*d_x_enc)
        x_enc = x_enc.permute(0, 2, 1, 3)               # (B, L, N, d_x_enc)
        x_flat = x_enc.reshape(B, L, N * self.d_x_enc)  # (B, L, N*d_x_enc)

        # --- Concatenate: [Y_1..Y_N, X̃_1..X̃_N] → (B, L, N*(1+k)) ---
        z = torch.cat([y_input, x_flat], dim=-1)

        # --- Add embeddings ---
        se = self.step_embed(t).unsqueeze(1)             # (B, 1, input_dim)
        te = self.temporal_embed[:L].unsqueeze(0)        # (1, L, input_dim)
        z = z + se + te

        # --- Project to d_model ---
        z = self.input_proj(z)                           # (B, L, d_model)

        # --- Self-attention layers ---
        for layer in self.layers:
            z = layer(z)

        # --- Output: d_model → N ---
        z = self.output_proj(z)                          # (B, L, N)
        return z.permute(0, 2, 1)                        # (B, N, L)


# ---------------------------------------------------------------------------
# MIDM with X conditioning
# ---------------------------------------------------------------------------

class MIDMCondX(nn.Module):
    """Full MIDM model with per-basin X conditioning."""

    def __init__(self, n_vars, max_seq_len, d_x, d_model=512, n_heads=8,
                 n_layers=4, n_diffusion_steps=50, cov_rank=8,
                 d_x_enc=4, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.cov = LowRankCovariance(n_vars, rank=cov_rank)
        self.eem = EEMCondX(
            n_vars, max_seq_len, d_x=d_x, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, n_diffusion_steps=n_diffusion_steps,
            d_x_enc=d_x_enc, dropout=dropout)

    def forward_diffusion(self, x0, t, schedule, noise=None):
        """Forward diffusion (same as pure MIDM)."""
        B, N, L = x0.shape
        device = x0.device
        if noise is None:
            noise = self.cov.sample_noise((B, L), device).permute(0, 2, 1)
        ab = schedule.alpha_bars[t].view(-1, 1, 1)
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
        return x_t, noise

    def predict_noise(self, x_t, x_prev_c, t, x_features):
        """Predict noise via EEM with X conditioning."""
        return self.eem(x_t, x_prev_c, t, x_features)


# ---------------------------------------------------------------------------
# Sampling with X conditioning
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_ddim_condx(model, schedule, y_obs, obs_idx, miss_idx,
                      x_features, n_steps=50):
    """
    DDIM sampling with conditional noise + X features.

    Args:
        model:      MIDMCondX
        schedule:   DiffusionSchedule
        y_obs:      (B, n_obs, L) — observed Y (clean, normalized)
        obs_idx:    (n_obs,) int
        miss_idx:   (n_miss,) int
        x_features: (B, N, L, D_x) — exogenous features for ALL basins
        n_steps:    reverse steps
    Returns:
        (B, n_miss, L) — imputed missing Y
    """
    model.eval()
    device = y_obs.device
    B, n_obs, L = y_obs.shape
    n_miss = len(miss_idx)
    N = model.n_vars
    total_T = schedule.n_steps

    # --- Initialize: conditional noise sampling ---
    ab_T = schedule.alpha_bars[-1]
    noise_obs = model.cov.sample_noise((B, L), device)
    noise_obs_dims = noise_obs[:, :, obs_idx]
    x_obs_T = (torch.sqrt(ab_T) * y_obs.permute(0, 2, 1)
               + torch.sqrt(1 - ab_T) * noise_obs_dims)

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

        if t_val > 0:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            ab_prev = schedule.alpha_bars[t_prev]
        else:
            ab_prev = torch.tensor(1.0, device=device)

        noise_for_obs = model.cov.sample_noise((B, L), device)[:, :, obs_idx]
        x_prev_c_obs = (torch.sqrt(ab_prev) * y_obs.permute(0, 2, 1)
                        + torch.sqrt(1 - ab_prev) * noise_for_obs)

        x_prev_c = torch.zeros(B, N, L, device=device)
        x_prev_c[:, obs_idx, :] = x_prev_c_obs.permute(0, 2, 1)

        # EEM with X conditioning
        eps = model.predict_noise(x_t, x_prev_c, t_batch, x_features)

        x0_pred = (x_t - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)
        x0_pred = x0_pred.clamp(-50, 50)
        x0_pred[:, obs_idx, :] = y_obs

        if i < len(timesteps) - 1:
            ab_next = schedule.alpha_bars[timesteps[i + 1]]
            x_t = torch.sqrt(ab_next) * x0_pred + torch.sqrt(1 - ab_next) * eps
        else:
            x_t = x0_pred

    return x_t[:, miss_idx, :]
