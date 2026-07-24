"""
Mamba: Linear-Time Sequence Modeling with Selective State Spaces
NeurIPS 2023

Key idea: State Space Model with selective (input-dependent) parameters.
Achieves linear complexity while maintaining the modeling power of Transformers.

Original paper: https://arxiv.org/abs/2312.00752

This is a simplified implementation focusing on the core selective SSM mechanism.

Adapted for X-to-Y (seq2seq) mode:
  Input:  (batch, seq_len, input_dim)
  Output: (batch, seq_len, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) - Core of Mamba

    State equation: h'(t) = A h(t) + B x(t)
    Output equation: y(t) = C h(t) + D x(t)

    Key: A, B, C are input-dependent (selective)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto'):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        if dt_rank == 'auto':
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        # Input projection: project to inner dimension * 2 (for x and z)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )

        # SSM parameters projection
        # Projects to: dt, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # dt (delta) projection
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize A (state matrix) - use log parameterization for stability
        # A is initialized as negative to ensure stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        @param x: (batch, seq_len, d_model)
        @return: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)

        # Convolution (local context)
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim padding
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)

        x = F.silu(x)

        # SSM parameters (selective - input-dependent)
        x_dbl = self.x_proj(x)  # (batch, seq_len, dt_rank + d_state * 2)

        # Split into dt, B, C
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )

        # Project dt
        dt = self.dt_proj(dt)  # (batch, seq_len, d_inner)
        dt = F.softplus(dt)  # Ensure positive

        # Get A from log parameterization
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Selective scan (simplified version)
        y = self.selective_scan(x, dt, A, B, C)

        # Gated output
        y = y * F.silu(z)

        # Output projection
        output = self.out_proj(y)

        return output

    def selective_scan(self, x, dt, A, B, C):
        """
        Selective scan algorithm (simplified sequential version)

        @param x: (batch, seq_len, d_inner)
        @param dt: (batch, seq_len, d_inner) - time step
        @param A: (d_inner, d_state) - state matrix
        @param B: (batch, seq_len, d_state) - input matrix
        @param C: (batch, seq_len, d_state) - output matrix
        @return: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A and B using dt
        # dA = exp(dt * A)
        # dB = dt * B
        dt = dt.unsqueeze(-1)  # (batch, seq_len, d_inner, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)

        dA = torch.exp(dt * A)  # (batch, seq_len, d_inner, d_state)

        # B: (batch, seq_len, d_state) -> (batch, seq_len, d_inner, d_state)
        B = B.unsqueeze(2).expand(-1, -1, d_inner, -1)
        dB = dt * B  # (batch, seq_len, d_inner, d_state)

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        # Sequential scan
        ys = []
        for t in range(seq_len):
            # State update: h = dA * h + dB * x
            h = dA[:, t] * h + dB[:, t] * x[:, t:t+1, :].transpose(1, 2)

            # Output: y = C * h + D * x
            # C[:, t]: (batch, d_state) -> expand to (batch, d_inner, d_state)
            # h: (batch, d_inner, d_state)
            # Sum over d_state dimension to get (batch, d_inner)
            C_t = C[:, t].unsqueeze(1)  # (batch, 1, d_state)
            y_t = (h * C_t).sum(dim=-1)  # (batch, d_inner)
            y_t = y_t + self.D * x[:, t]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (batch, seq_len, d_inner)

        return y


class MambaBlock(nn.Module):
    """Mamba block with residual connection and normalization"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(d_model, d_state, d_conv, expand)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class Mamba(nn.Module):
    """
    Mamba for X-to-Y time series prediction

    Uses selective state space models with linear complexity.
    """
    def __init__(self, input_dim, hidden_dim=64, adj_matrix=None,
                 num_layers=2, d_state=16, d_conv=4, expand=2,
                 recur_dropout=0, dropout=0.1, return_states=False,
                 device='cpu', seed=None):
        """
        @param input_dim: number of input features
        @param hidden_dim: model dimension (d_model)
        @param adj_matrix: [IGNORED] kept for API compatibility
        @param num_layers: number of Mamba blocks
        @param d_state: state dimension in SSM
        @param d_conv: convolution kernel size
        @param expand: expansion factor for inner dimension
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

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(hidden_dim, d_state, d_conv, expand)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, init_states=None):
        """
        @param x: (batch, seq_len, input_dim)
        @param init_states: [IGNORED]
        @return: (batch, seq_len, 1)
        """
        # Input projection
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)
            x = self.dropout(x)

        # Final normalization
        x = self.norm(x)

        # Output projection
        out = self.output_proj(x)  # (batch, seq_len, 1)

        return out
