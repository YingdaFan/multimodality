"""
DLinear: Are Transformers Effective for Time Series Forecasting?
AAAI 2023

Key idea: Simple linear layers on decomposed trend and seasonal components
can outperform complex Transformer models.

Original paper: https://arxiv.org/abs/2205.13504

Adapted for X-to-Y (seq2seq) mode:
  Input:  (batch, seq_len, input_dim)
  Output: (batch, seq_len, 1)
"""

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving average block to extract trend component"""
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        # Pad on both ends for same output length
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # AvgPool1d expects (batch, channels, seq_len)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """Series decomposition block: trend + seasonal"""
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        # x: (batch, seq_len, channels)
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear for X-to-Y time series prediction

    Decomposes input into trend and seasonal components,
    applies separate linear transformations, then combines.
    """
    def __init__(self, input_dim, hidden_dim=None, adj_matrix=None,
                 seq_len=365, kernel_size=25, individual=False,
                 recur_dropout=0, dropout=0, return_states=False,
                 device='cpu', seed=None):
        """
        @param input_dim: number of input features
        @param hidden_dim: [IGNORED] kept for API compatibility
        @param adj_matrix: [IGNORED] kept for API compatibility
        @param seq_len: sequence length (default 365 for yearly data)
        @param kernel_size: kernel size for moving average decomposition
        @param individual: if True, use separate linear for each feature
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
        self.hidden_size = hidden_dim if hidden_dim else input_dim
        self.seq_len = seq_len
        self.individual = individual

        # Series decomposition
        self.decomposition = SeriesDecomp(kernel_size)

        if self.individual:
            # Separate linear for each input feature
            self.Linear_Seasonal = nn.ModuleList([
                nn.Linear(seq_len, seq_len) for _ in range(input_dim)
            ])
            self.Linear_Trend = nn.ModuleList([
                nn.Linear(seq_len, seq_len) for _ in range(input_dim)
            ])
        else:
            # Shared linear across features
            self.Linear_Seasonal = nn.Linear(seq_len, seq_len)
            self.Linear_Trend = nn.Linear(seq_len, seq_len)

        # Output projection: from input_dim to 1
        self.output_proj = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        if self.individual:
            for layer in self.Linear_Seasonal:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            for layer in self.Linear_Trend:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        else:
            nn.init.xavier_uniform_(self.Linear_Seasonal.weight)
            nn.init.zeros_(self.Linear_Seasonal.bias)
            nn.init.xavier_uniform_(self.Linear_Trend.weight)
            nn.init.zeros_(self.Linear_Trend.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, init_states=None):
        """
        @param x: (batch, seq_len, input_dim)
        @param init_states: [IGNORED]
        @return: (batch, seq_len, 1)
        """
        # Decompose into seasonal and trend
        seasonal, trend = self.decomposition(x)
        # seasonal, trend: (batch, seq_len, input_dim)

        if self.individual:
            # Process each feature separately
            seasonal_out = torch.zeros_like(seasonal)
            trend_out = torch.zeros_like(trend)
            for i in range(self.input_dim):
                # Linear operates on seq_len dimension
                seasonal_out[:, :, i] = self.Linear_Seasonal[i](seasonal[:, :, i].clone())
                trend_out[:, :, i] = self.Linear_Trend[i](trend[:, :, i].clone())
        else:
            # Shared linear: transpose to (batch, input_dim, seq_len)
            seasonal_out = self.Linear_Seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            trend_out = self.Linear_Trend(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # Combine seasonal and trend
        out = seasonal_out + trend_out  # (batch, seq_len, input_dim)

        # Apply dropout
        out = self.dropout(out)

        # Project to output dimension
        out = self.output_proj(out)  # (batch, seq_len, 1)

        return out
