"""
FMDiff — backbone for Stochastic Interpolant calibration (sical_gx_enc.py).

Mirrors NsDiff.py exactly except its forward accepts an `apply_sigma_softplus`
flag that is forwarded to the underlying ConditionalGuidedModel. With
apply_sigma_softplus=False the second output is the raw linear projection,
which SI uses as the denoiser eta(t, x) = E[z | x_t = x].

NsDiff.py is left untouched so diffcal/fmcal continue to behave identically.
"""

import torch
import torch.nn as nn
from torch_timeseries.nn.embedding import DataEmbedding
import yaml
import argparse

from src.nn.tmdm_diffusion_utils import *
from src.layer.fmdiffdenoise import ConditionalGuidedModel
from src.utils.sigma import wv_sigma
from src.nn.wave_fusion import WaveSuperposition, WaveScatterFusion, WaveScatterInterference


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def compute_gx_term(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    gx_term = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[:t + 1].flip(dims=[0])
        cprod = torch.cat([torch.tensor([1]).to(slice_t.device),
                           torch.cumprod(slice_t, dim=0)])
        cprod = cprod[:-1] * ((1 - slice_t) ** 2)
        gx_term[t] = cprod.sum()
    return gx_term


def compute_tilde_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    tilde_alpha = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[:t + 1].flip(dims=[0])
        cprod = torch.cumprod(slice_t, dim=0)
        tilde_alpha[t] = cprod.sum()
    return tilde_alpha


def compute_hat_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    hat_alpha = torch.zeros_like(alpha)
    for t in range(n):
        slice_t = alpha[:t + 1].flip(dims=[0])
        cprod = torch.cumprod(slice_t, dim=0)
        cprod = cprod * slice_t
        hat_alpha[t] = cprod.sum()
    return hat_alpha


class FMDiff(nn.Module):
    """Backbone for Stochastic Interpolant calibration.

    Architecturally identical to NsDiff. The only difference is that forward
    accepts an `apply_sigma_softplus` flag (default True). With False, the
    second output of ConditionalGuidedModel is the raw linear projection
    (allowed to be negative), so SI can use this head to predict
    eta(t, x) = E[z | x_t = x].

    The schedule machinery (alphas, betas_tilde, betas_bar, etc.) is kept
    intact to preserve compatibility with existing checkpoint structures
    and time-embedding indexing.
    """

    def __init__(self, configs, device):
        super(FMDiff, self).__init__()

        self.args = configs
        self.device = device

        self.num_timesteps = configs.timesteps
        self.dataset_object = None
        betas = make_beta_schedule(
            schedule=configs.beta_schedule,
            num_timesteps=configs.timesteps,
            start=configs.beta_start,
            end=configs.beta_end,
        )
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)

        self.betas_bar = 1 - self.alphas_cumprod
        self.alphas_cumprod_sum = compute_tilde_alpha(alphas)

        self.alphas_tilde = self.alphas_cumprod_sum
        self.alphas_hat = compute_hat_alpha(alphas).to(self.device)
        self.betas_tilde = self.alphas_tilde - self.alphas_hat
        self.gx_term = compute_gx_term(alphas).to(self.device)

        assert (torch.tensor(self.betas_tilde) >= 0).all()
        assert ((self.betas_bar - self.betas_tilde) >= 0).all()

        self.betas_tilde_m_1 = torch.cat(
            [torch.ones(1, device=self.device), self.betas_tilde[:-1]], dim=0)
        self.betas_bar_m_1 = torch.cat(
            [torch.ones(1, device=self.device), self.betas_bar[:-1]], dim=0)

        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if configs.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_sum_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod_sum[:-1]], dim=0)

        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coeff_2 = (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_variance = posterior_variance

        self.tau = None

        c_out = getattr(configs, 'c_out', configs.enc_in)
        d_model = configs.CART_input_x_embed_dim

        # ConditionalGuidedModel here is from fmdiffdenoise (SI variant)
        self.diffussion_model = ConditionalGuidedModel(
            configs.timesteps, configs.enc_in, c_out, d_model=d_model,
        )

        time_embed = getattr(configs, 'time_embed', True)
        self.enc_embedding = DataEmbedding(
            configs.enc_in, configs.CART_input_x_embed_dim, configs.embed,
            configs.freq, configs.dropout, time_embed=time_embed,
        )

        # Bidirectional Wave Fusion (same as NsDiff)
        self.fusion_type = getattr(configs, 'fusion_type', None)
        seq_len = configs.seq_len

        if self.fusion_type and self.fusion_type in (
                'interference', 'scatter', 'scatterinterference'):
            if self.fusion_type == 'interference':
                self.wave_fusion = WaveSuperposition(d_model, seq_len)
            elif self.fusion_type == 'scatter':
                self.wave_fusion = WaveScatterFusion(
                    d_model, n_heads=4, dropout=configs.dropout)
            elif self.fusion_type == 'scatterinterference':
                self.wave_fusion = WaveScatterInterference(
                    d_model, seq_len, n_heads=4, dropout=configs.dropout)

            print(f"[FMDiff] Bidirectional mode enabled with {self.fusion_type} fusion")
        else:
            self.wave_fusion = None

    def forward(self, x, x_mark, y_t, y_0_hat, gx, t, apply_sigma_softplus=True):
        """
        Args:
            x:       (B, T, enc_in)
            x_mark:  (B, T, time_features)
            y_t:     (B, O, c_out) — current state of Y in the bridge
            y_0_hat: (B, O, c_out) — informed prior (LSTM output)
            gx:      (B, O, c_out) — variance slot (ones in current pipeline)
            t:       discrete timestep tensor
            apply_sigma_softplus:
                True  -> NsDiff-compatible behavior (second output passes softplus)
                False -> SI behavior (second output is raw, can be negative for eta)

        Returns:
            head1: velocity b (SI) or eps_pred (NsDiff style)
            head2: eta (SI, raw) or sigma_theta (NsDiff style, positive)
        """
        if self.wave_fusion is not None:
            enc_fwd = self.enc_embedding(x, x_mark)
            x_flip = x.flip(dims=[1])
            x_mark_flip = x_mark.flip(dims=[1])
            enc_bwd_flip = self.enc_embedding(x_flip, x_mark_flip)
            enc_bwd = enc_bwd_flip.flip(dims=[1])
            enc_out = self.wave_fusion(enc_fwd, enc_bwd)
        else:
            enc_out = self.enc_embedding(x, x_mark)

        head1, head2 = self.diffussion_model(
            enc_out, y_t, y_0_hat, gx, t,
            apply_sigma_softplus=apply_sigma_softplus,
        )

        return head1, head2
