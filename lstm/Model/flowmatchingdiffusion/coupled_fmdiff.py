"""
Coupled FM + Diffusion model adapted for the imputation pipeline.

Adapted from flowmatchingdiffusion.zip (coupled_streamflow_model.py) with
the following changes for our pipeline:

  1. NaN-mask in `loss()`:  y_obs has lots of NaN gaps; without masking
     the entire training loss becomes NaN within one step. We follow the
     `rmse_masked` pattern from `torch_utils.py`:
       - replace NaN in y0 with 0 BEFORE perturbation (so y_t / forward
         pass do not propagate NaN)
       - aggregate loss only at non-NaN positions

  2. LSTM-compatible nn.Module API:
       - __init__ signature mirrors `model.LSTM` (input_dim, hidden_dim,
         adj_matrix, dropout, device, seed) so that base.py can swap in
         this model with no other code changes
       - forward(x) -> sampled prediction (B, L, 1) for use by
         `predict_torch` / `val_loop` (when configured to use forward)
       - compute_loss(x, y_true) -> scalar training loss; the training
         loop in `torch_utils.py` is dispatched to call this when the
         model exposes it

Theoretical core (unchanged from the zip):
  - VP-SDE forward perturbation
  - Closed-form FM target (no autograd over schedule)
  - Two independent networks (score_net + vel_net) coupled ONLY through
    the ODE consistency loss u = f - 0.5 g^2 s
  - Reverse-time integration with positive dt
"""

import torch
import torch.nn as nn

from .dlinear_ts import DLinearBackboneTS
from .transformer_backbone_ts import TransformerBackboneTS


# Backbone registry: name -> class.  Use to swap implementations without
# touching the coupling logic.
_BACKBONES = {
    'dlinear': DLinearBackboneTS,
    'transformer': TransformerBackboneTS,
}


def _build_backbone(name, cov_dim, target_dim, seq_len, dropout, **kwargs):
    """
    Construct one backbone instance.  All backbones expose
    `forward(x_cov, y_t, t) -> (B, L, target_dim)` so they are
    interchangeable; only construction kwargs differ.
    """
    if name not in _BACKBONES:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose from {list(_BACKBONES)}"
        )
    return _BACKBONES[name](
        cov_dim=cov_dim, target_dim=target_dim, seq_len=seq_len,
        dropout=dropout, **kwargs,
    )


# ---------------------------------------------------------------------------
# VP-SDE schedule (closed-form derivatives)
# ---------------------------------------------------------------------------
class VPSchedule:
    def __init__(self, beta_min=0.1, beta_max=20.0):
        self.b_min = beta_min
        self.b_max = beta_max

    def _b_view(self, t):
        # (B,) -> (B, 1, 1) for broadcasting against (B, L, C)
        return (self.b_min + t * (self.b_max - self.b_min)).view(-1, 1, 1)

    def beta(self, t):
        return self._b_view(t)

    def A(self, t):
        return (
            self.b_min * t + 0.5 * (self.b_max - self.b_min) * t.pow(2)
        ).view(-1, 1, 1)

    def alpha(self, t):
        return torch.exp(-0.5 * self.A(t))

    def sigma(self, t):
        return torch.sqrt((1 - torch.exp(-self.A(t))).clamp(min=1e-8))

    def alpha_dot(self, t):
        return -0.5 * self.beta(t) * self.alpha(t)

    def sigma_dot(self, t):
        return self.beta(t) * torch.exp(-self.A(t)) / (2 * self.sigma(t))

    def f(self, x, t):
        return -0.5 * self.beta(t) * x

    def g_sq(self, t):
        return self.beta(t)


# ---------------------------------------------------------------------------
# Coupled FM + Diffusion core
# ---------------------------------------------------------------------------
class _CoupledCore(nn.Module):
    """
    Two DLinear-based networks: score_net and vel_net.
    Trained on the same forward perturbation, coupled ONLY through the
    ODE consistency loss. No cross-conditioning in the forward pass.
    """

    def __init__(
        self,
        cov_dim,
        seq_len,
        backbone='dlinear',
        backbone_kwargs=None,
        # legacy DLinear kwargs (kept so existing callers don't break)
        kernel_size=25,
        time_embed_dim=64,
        individual=False,
        dropout=0.0,
        share_backbone=False,
        sde=None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.cov_dim = cov_dim
        self.share_backbone = share_backbone
        self.sde = sde or VPSchedule()
        self.backbone = backbone

        # Default kwargs per backbone
        if backbone_kwargs is None:
            if backbone == 'dlinear':
                backbone_kwargs = dict(
                    kernel_size=kernel_size,
                    time_embed_dim=time_embed_dim,
                    individual=individual,
                )
            elif backbone == 'transformer':
                backbone_kwargs = dict(
                    d_model=128, n_heads=8, n_blocks=4,
                    time_embed_dim=128, ffn_mult=4,
                )

        common = dict(
            cov_dim=cov_dim, target_dim=1, seq_len=seq_len, dropout=dropout,
        )

        if share_backbone:
            self.trunk = _build_backbone(backbone, **common, **backbone_kwargs)
            self.score_head = nn.Sequential(
                nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1)
            )
            self.vel_head = nn.Sequential(
                nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1)
            )
        else:
            self.score_net = _build_backbone(backbone, **common, **backbone_kwargs)
            self.vel_net = _build_backbone(backbone, **common, **backbone_kwargs)

    def predict_score(self, x_cov, y_t, t):
        if self.share_backbone:
            h = self.trunk(x_cov, y_t, t)
            return self.score_head(h)
        return self.score_net(x_cov, y_t, t)

    def predict_velocity(self, x_cov, y_t, t):
        if self.share_backbone:
            h = self.trunk(x_cov, y_t, t)
            return self.vel_head(h)
        return self.vel_net(x_cov, y_t, t)

    def perturb(self, y0, t):
        eps = torch.randn_like(y0)
        alpha = self.sde.alpha(t)
        sigma = self.sde.sigma(t)
        return alpha * y0 + sigma * eps, eps, alpha, sigma

    def fm_target(self, y0, eps, t):
        return self.sde.alpha_dot(t) * y0 + self.sde.sigma_dot(t) * eps

    @torch.no_grad()
    def sample(self, x_cov, num_steps=50, use="vel", t_min=1e-2):
        """
        Args:
            x_cov:     (B, L, cov_dim)
            num_steps: integration steps
            use:       'vel' | 'score' | 'avg'
        Returns:
            (B, L, 1)
        """
        B, L, _ = x_cov.shape
        device = x_cov.device

        ts = torch.linspace(1.0 - t_min, t_min, num_steps + 1, device=device)
        x = torch.randn(B, L, 1, device=device)

        for i in range(num_steps):
            t_now = ts[i].expand(B)
            dtau = (ts[i] - ts[i + 1]).item()  # positive

            if use == "vel":
                u = self.predict_velocity(x_cov, x, t_now)
                dx = -u
            elif use == "score":
                s = self.predict_score(x_cov, x, t_now)
                f_xt = self.sde.f(x, t_now)
                g_sq = self.sde.g_sq(t_now)
                dx = -(f_xt - 0.5 * g_sq * s)
            elif use == "avg":
                u = self.predict_velocity(x_cov, x, t_now)
                s = self.predict_score(x_cov, x, t_now)
                f_xt = self.sde.f(x, t_now)
                g_sq = self.sde.g_sq(t_now)
                dx = 0.5 * (-u + -(f_xt - 0.5 * g_sq * s))
            else:
                raise ValueError(use)

            x = x + dtau * dx

        return x


# ---------------------------------------------------------------------------
# LSTM-compatible wrapper
# ---------------------------------------------------------------------------
class CoupledFMDiff(nn.Module):
    """
    Drop-in replacement for `lstm.model.LSTM` in the imputation Stage 1.

    API:
      __init__(input_dim, hidden_dim=None, adj_matrix=None, dropout=0.0,
               device='cpu', seed=None, seq_len=168, ...)
        - hidden_dim, adj_matrix: ignored, kept for signature compatibility
        - other kwargs are FM/Diff-specific knobs

      forward(x):  (B, L, F) -> (B, L, 1)  via reverse-time ODE sampling
        Used by predict_torch.

      compute_loss(x, y_true):  (B, L, F), (B, L, 1) -> scalar
        Used by torch_utils.train_loop / val_loop when present.

    NaN handling matches `rmse_masked`:
      - any cell where y_true is NaN is filled with 0 before perturbation
      - score / vel / ode losses are masked to non-NaN positions
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        adj_matrix=None,
        dropout=0.0,
        device="cpu",
        seed=None,
        seq_len=168,
        # backbone selection
        backbone='dlinear',                # 'dlinear' | 'transformer'
        backbone_kwargs=None,              # explicit overrides for backbone
        # legacy DLinear knobs (only used when backbone='dlinear' and
        # backbone_kwargs is None)
        kernel_size=25,
        time_embed_dim=64,
        individual=False,
        share_backbone=False,
        # FM/Diff knobs
        lambda_ode=1.0,
        sample_steps=50,
        sample_method="vel",
        t_eps=1e-2,
        beta_min=0.1,
        beta_max=20.0,
        return_states=False,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # accept-but-ignore (LSTM API parity)
        self.adj_matrix = adj_matrix
        self.return_states = return_states

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.lambda_ode = lambda_ode
        self.sample_steps = sample_steps
        self.sample_method = sample_method
        self.t_eps = t_eps
        self.backbone = backbone

        sde = VPSchedule(beta_min=beta_min, beta_max=beta_max)
        self.core = _CoupledCore(
            cov_dim=input_dim,
            seq_len=seq_len,
            backbone=backbone,
            backbone_kwargs=backbone_kwargs,
            kernel_size=kernel_size,
            time_embed_dim=time_embed_dim,
            individual=individual,
            dropout=dropout,
            share_backbone=share_backbone,
            sde=sde,
        )

    # ------------------------------------------------------------------
    # Loss with NaN masking
    # ------------------------------------------------------------------
    def compute_loss(self, x, y_true, lambda_ode=None, return_components=False):
        """
        Args:
            x:                 (B, L, F) covariates
            y_true:            (B, L, 1) targets, may contain NaN
            lambda_ode:        if given, override self.lambda_ode (use this
                               when the trainer wants to ramp up over steps)
            return_components: if True, also return a dict with score/vel/ode
                               sub-losses for logging.

        Returns:
            scalar (with grad), or (scalar, dict) if return_components.
        """
        # mask = True where y is observed (non-NaN)
        mask = ~torch.isnan(y_true)                          # (B, L, 1)
        y0 = torch.where(mask, y_true, torch.zeros_like(y_true))

        n_valid = mask.sum().clamp(min=1).to(y0.dtype)

        B = y0.shape[0]
        device = y0.device

        # sample diffusion timestep
        t = torch.rand(B, device=device).clamp(self.t_eps, 1.0 - self.t_eps)

        y_t, eps, alpha, sigma = self.core.perturb(y0, t)

        # -------- score loss (masked) --------
        # target s* = -eps / sigma; weight by sigma^2 ~ noise-pred MSE
        s_pred = self.core.predict_score(x, y_t, t)
        s_target = -eps / sigma
        sq_score = (s_pred - s_target).pow(2) * sigma.pow(2)     # (B, L, 1)
        score_loss = (sq_score * mask).sum() / n_valid

        # -------- velocity loss (masked) --------
        u_pred = self.core.predict_velocity(x, y_t, t)
        u_target = self.core.fm_target(y0, eps, t)
        sq_vel = (u_pred - u_target).pow(2)                       # (B, L, 1)
        vel_loss = (sq_vel * mask).sum() / n_valid

        # -------- ODE consistency loss (masked) --------
        # u = f - 0.5 g^2 s  must hold pointwise. Restrict to observed
        # positions to match the supervised loss footprint.
        f_yt = self.core.sde.f(y_t, t)
        g_sq = self.core.sde.g_sq(t)
        residual = u_pred - f_yt + 0.5 * g_sq * s_pred             # (B, L, 1)
        weight = 1.0 / (1.0 + g_sq.pow(2))                         # (B, 1, 1)
        ode_loss = ((residual.pow(2) * weight) * mask).sum() / n_valid

        lam = self.lambda_ode if lambda_ode is None else lambda_ode
        total = score_loss + vel_loss + lam * ode_loss

        # if no valid positions in entire batch, return zero (matches rmse_masked)
        if mask.sum() == 0:
            total = torch.zeros((), device=device, dtype=y0.dtype, requires_grad=True)

        if return_components:
            return total, {
                'score': float(score_loss.detach()),
                'vel': float(vel_loss.detach()),
                'ode': float(ode_loss.detach()),
                'total': float(total.detach()),
                'lambda_ode': float(lam),
                'n_valid': int(mask.sum()),
            }
        return total

    # ------------------------------------------------------------------
    # Inference: forward(x) -> sampled prediction
    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: (B, L, F)
        Returns:
            (B, L, 1) predicted streamflow (in normalized space, same as
            LSTM's forward output)
        """
        with torch.no_grad():
            y_pred = self.core.sample(
                x_cov=x,
                num_steps=self.sample_steps,
                use=self.sample_method,
                t_min=self.t_eps,
            )
        if self.return_states:
            return y_pred, (None, None)
        return y_pred

    # ------------------------------------------------------------------
    # Optional: ensemble sampling for uncertainty quantification
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_ensemble(self, x, n_samples=20):
        out = [self.forward(x) for _ in range(n_samples)]
        return torch.stack(out, dim=0)  # (n_samples, B, L, 1)
