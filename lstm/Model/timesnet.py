"""
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
ICLR 2023

Key idea: Transform 1D time series into 2D space using FFT to discover
periodic patterns, then apply 2D convolution (Inception blocks).

Original paper: https://arxiv.org/abs/2210.02186

Adapted for X-to-Y (seq2seq) mode:
  Input:  (batch, seq_len, input_dim)
  Output: (batch, seq_len, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def FFT_for_Period(x, k=2):
    """
    Find top-k dominant periods using FFT

    @param x: (batch, seq_len, channels)
    @param k: number of top periods to return
    @return: periods, weights for top-k frequencies
    """
    # FFT along time dimension
    xf = torch.fft.rfft(x, dim=1)

    # Compute amplitude spectrum
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # Remove DC component

    # Find top-k frequencies
    _, top_indices = torch.topk(frequency_list, k)
    top_indices = top_indices.detach().cpu().numpy()

    # Convert frequency indices to periods
    period = x.shape[1] // top_indices

    return period, abs(xf).mean(-1)[:, top_indices]


class InceptionBlock_V1(nn.Module):
    """Inception block for 2D convolution"""
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        # Multiple kernel sizes for multi-scale pattern extraction
        kernels = []
        for i in range(num_kernels):
            kernel_size = 2 * i + 1  # 1, 3, 5, 7, ...
            padding = i  # same padding
            kernels.append(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=(kernel_size, 1),
                         padding=(padding, 0))
            )
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x):
        # x: (batch, in_channels, period, num_periods)
        res_list = []
        for kernel in self.kernels:
            res_list.append(kernel(x))

        # Average outputs from all kernel sizes
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    """Single TimesNet block"""
    def __init__(self, seq_len, d_model, d_ff, top_k=5, num_kernels=6):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k

        # 2D convolution for period-based modeling
        self.conv = nn.Sequential(
            InceptionBlock_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        """
        @param x: (batch, seq_len, d_model)
        @return: (batch, seq_len, d_model)
        """
        batch_size, seq_len, n_vars = x.size()

        # Find top-k periods using FFT
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        res = []
        for i, period in enumerate(period_list):
            # Handle edge case
            if period <= 0 or seq_len % period != 0:
                period = seq_len // (seq_len // max(period, 1)) if period > 0 else seq_len

            # Reshape to 2D: (batch, d_model, period, num_periods)
            if seq_len % period == 0:
                # Exact division
                out = x.reshape(batch_size, period, seq_len // period, n_vars)
                out = out.permute(0, 3, 1, 2).contiguous()  # (batch, d_model, period, num_periods)
            else:
                # Pad to make divisible
                pad_len = period - (seq_len % period)
                x_pad = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
                out = x_pad.reshape(batch_size, period, -1, n_vars)
                out = out.permute(0, 3, 1, 2).contiguous()

            # Apply 2D convolution
            out = self.conv(out)

            # Reshape back to 1D
            out = out.permute(0, 2, 3, 1).contiguous()  # (batch, period, num_periods, d_model)
            out = out.reshape(batch_size, -1, n_vars)  # (batch, period*num_periods, d_model)

            # Truncate to original length
            out = out[:, :seq_len, :]
            res.append(out)

        # Weighted sum based on FFT amplitudes
        res = torch.stack(res, dim=-1)  # (batch, seq_len, d_model, top_k)
        period_weight = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * period_weight, dim=-1)  # (batch, seq_len, d_model)

        return res


class TimesNet(nn.Module):
    """
    TimesNet for X-to-Y time series prediction

    Transforms 1D time series to 2D representation based on
    discovered periods, applies 2D convolution, then converts back.
    """
    def __init__(self, input_dim, hidden_dim=64, adj_matrix=None,
                 seq_len=365, num_layers=2, top_k=5, num_kernels=6,
                 d_ff=None, recur_dropout=0, dropout=0.1,
                 return_states=False, device='cpu', seed=None):
        """
        @param input_dim: number of input features
        @param hidden_dim: model dimension (d_model)
        @param adj_matrix: [IGNORED] kept for API compatibility
        @param seq_len: sequence length
        @param num_layers: number of TimesBlock layers
        @param top_k: number of top periods to use
        @param num_kernels: number of kernels in Inception block
        @param d_ff: feedforward dimension (default: 2 * hidden_dim)
        @param recur_dropout: [IGNORED]
        @param dropout: dropout rate
        @param return_states: [IGNORED]
        @param device: cpu or cuda
        @param seed: random seed
        """
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers

        if d_ff is None:
            d_ff = 2 * hidden_dim

        # Input embedding
        self.enc_embedding = nn.Linear(input_dim, hidden_dim)

        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, hidden_dim, d_ff, top_k, num_kernels)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.zeros_(self.enc_embedding.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, init_states=None):
        """
        @param x: (batch, seq_len, input_dim)
        @param init_states: [IGNORED]
        @return: (batch, seq_len, 1)
        """
        # Input embedding
        enc_out = self.enc_embedding(x)  # (batch, seq_len, hidden_dim)

        # Apply TimesNet blocks with residual connections
        for i, (block, norm) in enumerate(zip(self.blocks, self.layer_norms)):
            enc_out = norm(block(enc_out) + enc_out)
            enc_out = self.dropout(enc_out)

        # Output projection
        out = self.output_proj(enc_out)  # (batch, seq_len, 1)

        return out
