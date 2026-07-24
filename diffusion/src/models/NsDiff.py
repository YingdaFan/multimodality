import torch
import torch.nn as nn
from torch_timeseries.nn.embedding import DataEmbedding
import yaml
import argparse
from src.nn.tmdm_diffusion_utils import *
from src.layer.denoise import ConditionalGuidedModel
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
        slice_t = alpha[:t+1].flip(dims=[0]) # at, at-1, at-2, a1
        cprod = torch.cat([torch.tensor([1]).to(slice_t.device), torch.cumprod(slice_t, dim=0)])
        cprod = cprod[:-1] * ((1 - slice_t)**2) # at^2, at-1^2*at, at-1*at-2^2*at, ...
        gx_term[t] = cprod.sum()
    return gx_term


def compute_tilde_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    tilde_alpha = torch.zeros_like(alpha)  
    
    for t in range(n):
        slice_t = alpha[:t+1].flip(dims=[0])
        cprod = torch.cumprod(slice_t, dim=0)
        tilde_alpha[t] = cprod.sum()
    return tilde_alpha

def compute_hat_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha = alpha.float()
    n = alpha.shape[0]
    hat_alpha = torch.zeros_like(alpha)  
    for t in range(n):
        slice_t = alpha[:t+1].flip(dims=[0]) # at, at-1, at-2, ...
        cprod = torch.cumprod(slice_t, dim=0) # at, at-1*at, at-1*at-2*at, ...
        cprod = cprod * slice_t # at^2, at-1^2*at, at-1*at-2^2*at, ...
        hat_alpha[t] = cprod.sum()
    return hat_alpha

    # n = len(alpha)
    # if n == 0:
    #     return []
    # res = [0]*n
    # res[0] = alpha[0]**2
    # for t in range(1, n):
    #     res[t] = alpha[t]**2 + alpha[t]*res[t-1]
    # return torch.tensor(res)

# def compute_tilde_alpha(alpha):
#     n = len(alpha)
#     tilde_alpha = np.zeros(n, dtype=float)
#     for t in range(n):
#         import pdb;pdb.set_trace()
#         slice_t = alpha[t::-1] 
#         cprod = np.cumprod(slice_t)
#         tilde_alpha[t] = cprod.sum()
    
#     return tilde_alpha

class NsDiff(nn.Module):
    """
    Vanilla Transformer
    """

    def __init__(self, configs, device):
        super(NsDiff, self).__init__()

        self.args = configs
        self.device = device

        # self.model_var_type = configs.var_type
        self.num_timesteps = configs.timesteps
        self.dataset_object = None
        betas = make_beta_schedule(schedule=configs.beta_schedule, num_timesteps=configs.timesteps,
                                   start=configs.beta_start, end=configs.beta_end)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        
        self.betas_bar = 1 - self.alphas_cumprod
        # self.alphas_cumprod_sum = torch.cumsum(alphas_cumprod.flip(0), dim=0).flip(0)
        self.alphas_cumprod_sum = compute_tilde_alpha(alphas)
        
        self.alphas_tilde = self.alphas_cumprod_sum
        self.alphas_hat = compute_hat_alpha(alphas).to(self.device)
        self.betas_tilde = self.alphas_tilde  - self.alphas_hat
        self.gx_term = compute_gx_term(alphas).to(self.device) # full compute to avoid precision issue
        # import pdb;pdb.set_trace()
        assert (torch.tensor(self.betas_tilde) >= 0).all()
        # import pdb;pdb.set_trace(), ((self.betas_bar - self.betas_tilde)[((self.betas_bar - self.betas_tilde)>=0)])
        assert ((self.betas_bar - self.betas_tilde)>=0).all()
        # (self.betas_bar - self.betas_tilde)[((self.betas_bar - self.betas_tilde)>0)]
        
        
        self.betas_tilde_m_1 = torch.cat(
            [torch.ones(1, device=self.device), self.betas_tilde[:-1]], dim=0
        )
        self.betas_bar_m_1 = torch.cat(
            [torch.ones(1, device=self.device), self.betas_bar[:-1]], dim=0
        )

        
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        if configs.beta_schedule == "cosine":
            self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_sum_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod_sum[:-1]], dim=0
        )

        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance
        # if self.model_var_type == "fixedlarge":
        #     self.logvar = betas.log()
        #     # torch.cat(
        #     # [posterior_variance[1:2], betas[1:]], dim=0).log()
        # elif self.model_var_type == "fixedsmall":
        #     self.logvar = posterior_variance.clamp(min=1e-20).log()

        self.tau = None  # precision fo test NLL computation

        # Get output dimension (c_out), defaults to enc_in (backward compatible)
        c_out = getattr(configs, 'c_out', configs.enc_in)
        d_model = configs.CART_input_x_embed_dim  # encoder embedding dimension

        # CATE MLP - supports enc_in != c_out (MS mode), with X feature conditioning
        self.diffussion_model = ConditionalGuidedModel(
            configs.timesteps, configs.enc_in, c_out, d_model=d_model
        )

        # Support disabling time embedding (time_embed parameter)
        time_embed = getattr(configs, 'time_embed', True)  # default True for backward compatibility
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.CART_input_x_embed_dim, configs.embed, configs.freq,
                                           configs.dropout, time_embed=time_embed)

        # ========================================
        # Bidirectional Wave Fusion
        # fusion_type: None=unidirectional, 'interference'/'scatter'/'scatterinterference'=bidirectional
        # ========================================
        self.fusion_type = getattr(configs, 'fusion_type', None)
        seq_len = configs.seq_len

        if self.fusion_type and self.fusion_type in ('interference', 'scatter', 'scatterinterference'):
            # Create wave fusion module based on fusion_type
            if self.fusion_type == 'interference':
                self.wave_fusion = WaveSuperposition(d_model, seq_len)
            elif self.fusion_type == 'scatter':
                self.wave_fusion = WaveScatterFusion(d_model, n_heads=4, dropout=configs.dropout)
            elif self.fusion_type == 'scatterinterference':
                self.wave_fusion = WaveScatterInterference(d_model, seq_len, n_heads=4, dropout=configs.dropout)

            print(f"[NsDiff] Bidirectional mode enabled with {self.fusion_type} fusion")
        else:
            self.wave_fusion = None

    def forward(self, x, x_mark, y_t, y_0_hat, gx, t):
        # ========================================
        # Bidirectional Encoding with Wave Fusion
        # ========================================
        if self.wave_fusion is not None:
            # Forward pass: encode in original order
            enc_fwd = self.enc_embedding(x, x_mark)  # (B, T, d_model)

            # Backward pass: flip input, encode, flip output back
            x_flip = x.flip(dims=[1])
            x_mark_flip = x_mark.flip(dims=[1])
            enc_bwd_flip = self.enc_embedding(x_flip, x_mark_flip)
            enc_bwd = enc_bwd_flip.flip(dims=[1])  # Flip back to original order

            # Wave fusion: combine forward and backward representations
            enc_out = self.wave_fusion(enc_fwd, enc_bwd)  # (B, T, d_model)
        else:
            # Original unidirectional encoding
            enc_out = self.enc_embedding(x, x_mark)  # B, T, d_model

        dec_out, sigma = self.diffussion_model(enc_out, y_t, y_0_hat, gx, t)

        return dec_out, sigma
