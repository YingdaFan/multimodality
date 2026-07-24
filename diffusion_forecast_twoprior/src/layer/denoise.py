import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    """
    X Feature Condition Network - TimeGrad/TSDiff style

    Uses Cross-Attention to let the denoising process attend to conditional information:
    - Query: Y denoising state (to be predicted)
    - Key/Value: X conditional features (known driving information)

    Architecture:
    1. Local feature extraction: 1D convolution captures local patterns in the time dimension
    2. Cross-Attention: Y queries X, letting the denoising process attend to relevant X information
    3. Feature fusion: Fuses attention output with Y state
    """

    def __init__(self, d_model, y_state_dim, x_cond_dim=64, n_heads=4, dropout=0.1):
        """
        Args:
            d_model: encoder embedding dimension
            y_state_dim: dimension of Y state input (c_out * 2 for [y_t, y_0_hat])
            x_cond_dim: conditional vector output dimension
            n_heads: number of attention heads
            dropout: dropout rate
        """
        super(XConditionNetwork, self).__init__()
        self.d_model = d_model
        self.x_cond_dim = x_cond_dim

        # === Local feature extraction (temporal convolution) ===
        # Capture temporal patterns at different scales
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, x_cond_dim, kernel_size=3, padding=1),
        )
        self.local_norm = nn.LayerNorm(x_cond_dim)

        # === Cross-Attention ===
        # Y (denoising state) as Query, X (condition) as Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=x_cond_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(x_cond_dim)

        # === Y state projection ===
        self.y_proj = nn.Linear(y_state_dim, x_cond_dim)

        # === Output projection ===
        self.output_proj = nn.Sequential(
            nn.Linear(x_cond_dim, x_cond_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(x_cond_dim, x_cond_dim),
        )
        self.output_norm = nn.LayerNorm(x_cond_dim)

    def forward(self, x_enc, y_state):
        """
        Args:
            x_enc: encoder output (B, T, d_model) - condition features
            y_state: current state of Y (B, O, y_state_dim) - Y during denoising

        Returns:
            x_cond: condition vector (B, O, x_cond_dim)
        """
        B, T, _ = x_enc.shape
        O = y_state.shape[1]

        # === Local feature extraction ===
        # (B, T, d_model) -> (B, d_model, T) -> (B, x_cond_dim, T) -> (B, T, x_cond_dim)
        x_local = self.local_conv(x_enc.transpose(1, 2)).transpose(1, 2)
        x_local = self.local_norm(x_local)  # (B, T, x_cond_dim)

        # Keep full T dimension for cross-attention Key/Value so the denoiser
        # can attend to the entire history window (not just the last O steps).
        # Only interpolate when T < O (rare edge case).
        if T < O:
            x_local = F.interpolate(
                x_local.transpose(1, 2),
                size=O,
                mode='linear',
                align_corners=True
            ).transpose(1, 2)
        # x_local: (B, T, x_cond_dim) when T >= O, (B, O, x_cond_dim) when T < O

        # === Y state projection ===
        y_query = self.y_proj(y_state)  # (B, O, x_cond_dim)

        # === Cross-Attention ===
        # Query: Y state, Key/Value: X condition
        attn_out, _ = self.cross_attn(
            query=y_query,
            key=x_local,
            value=x_local
        )  # (B, O, x_cond_dim)
        attn_out = self.attn_norm(attn_out + y_query)  # residual connection

        # === Output fusion ===
        x_cond = self.output_proj(attn_out)
        x_cond = self.output_norm(x_cond + attn_out)  # residual connection

        return x_cond  # (B, O, x_cond_dim)


class ConditionalGuidedModel(nn.Module):
    def __init__(self, diff_steps, enc_in, c_out=None, d_model=32, x_cond_dim=64, n_heads=4, dropout=0.1):
        """
        Conditional guided diffusion model - TimeGrad/TSDiff style

        Uses Cross-Attention to inject conditional information:
        - Condition: [X, Y_history] via encoder embedding
        - Target: Y (1-dimensional)

        Args:
            diff_steps: number of diffusion steps
            enc_in: condition input dimension (X + Y_history, dynamically read from data)
            c_out: diffusion target dimension (Y = 1 dimension)
            d_model: encoder embedding dimension
            x_cond_dim: condition vector dimension
            n_heads: number of attention heads
            dropout: dropout rate
        """
        super(ConditionalGuidedModel, self).__init__()
        n_steps = diff_steps + 1

        if c_out is None:
            c_out = enc_in
        self.c_out = c_out
        self.enc_in = enc_in
        self.d_model = d_model
        self.x_cond_dim = x_cond_dim

        # === Enhanced condition network (Cross-Attention) ===
        self.x_condition_net = XConditionNetwork(
            d_model=d_model,
            y_state_dim=c_out * 2,  # [y_t, y_0_hat]
            x_cond_dim=x_cond_dim,
            n_heads=n_heads,
            dropout=dropout
        )

        # === Denoising MLP ===
        # Input: y_t(c_out) + y_0_hat(c_out) + g_x(c_out) + x_cond(x_cond_dim)
        data_dim = c_out * 3 + x_cond_dim
        hidden_dim = 256

        self.lin1 = ConditionalLinear(data_dim, hidden_dim, n_steps)
        self.lin2 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin3 = ConditionalLinear(hidden_dim, hidden_dim, n_steps)
        self.lin4 = ConditionalLinear(hidden_dim, 128, n_steps)

        self.out_proj = nn.Linear(128, c_out)
        self.sigma_lin = nn.Linear(128, c_out)

        # LayerNorm for stability
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y_t, y_0_hat, g_x, t):
        """
        Forward pass

        Args:
            x: encoder output (B, T, d_model) - embedding of condition features [X, Y_history]
            y_t: noisy sample at current timestep (B, O, c_out)
            y_0_hat: conditional prediction (B, O, c_out)
            g_x: variance estimate (B, O, c_out)
            t: diffusion timestep

        Returns:
            eps_pred: noise prediction (B, O, c_out)
            sigma: variance estimate (B, O, c_out)
        """
        # === Extract conditional information (Cross-Attention) ===
        # Use y_t + y_0_hat as the current state of Y
        y_state = torch.cat([y_t, y_0_hat], dim=-1)  # (B, O, c_out*2)
        x_cond = self.x_condition_net(x, y_state)  # (B, O, x_cond_dim)

        # === Concatenate all inputs ===
        h = torch.cat((y_t, y_0_hat, g_x, x_cond), dim=-1)  # (B, O, c_out*3 + x_cond_dim)

        # === Denoising MLP with Residual ===
        h1 = F.silu(self.lin1(h, t))
        h1 = self.norm1(h1)
        h1 = self.dropout(h1)

        h2 = F.silu(self.lin2(h1, t))
        h2 = self.norm2(h2)
        h2 = h2 + h1  # residual
        h2 = self.dropout(h2)

        h3 = F.silu(self.lin3(h2, t))
        h3 = self.norm3(h3)
        h3 = h3 + h2  # residual
        h3 = self.dropout(h3)

        h4 = F.silu(self.lin4(h3, t))

        # === Output ===
        eps_pred = self.out_proj(h4)
        sigma = F.softplus(self.sigma_lin(h4))

        return eps_pred, sigma


# # deterministic feed forward neural network
# class DeterministicFeedForwardNeuralNetwork(nn.Module):

#     def __init__(self, dim_in, dim_out, hid_layers,
#                  use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
#         super(DeterministicFeedForwardNeuralNetwork, self).__init__()
#         self.dim_in = dim_in  # dimension of nn input
#         self.dim_out = dim_out  # dimension of nn output
#         self.hid_layers = hid_layers  # nn hidden layer architecture
#         self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
#         self.use_batchnorm = use_batchnorm  # whether apply batch norm
#         self.negative_slope = negative_slope  # negative slope for LeakyReLU
#         self.dropout_rate = dropout_rate
#         layers = self.create_nn_layers()
#         self.network = nn.Sequential(*layers)

#     def create_nn_layers(self):
#         layers = []
#         for idx in range(len(self.nn_layers) - 1):
#             layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
#             if self.use_batchnorm:
#                 layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
#             layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
#             layers.append(nn.Dropout(p=self.dropout_rate))
#         layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
#         return layers

#     def forward(self, x):
#         return self.network(x)


# # early stopping scheme for hyperparameter tuning
# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""

#     def __init__(self, patience=10, delta=0):
#         """
#         Args:
#             patience (int): Number of steps to wait after average improvement is below certain threshold.
#                             Default: 10
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement;
#                            shall be a small positive value.
#                            Default: 0
#             best_score: value of the best metric on the validation set.
#             best_epoch: epoch with the best metric on the validation set.
#         """
#         self.patience = patience
#         self.delta = delta
#         self.counter = 0
#         self.best_score = None
#         self.best_epoch = None
#         self.early_stop = False

#     def __call__(self, val_cost, epoch, verbose=False):

#         score = val_cost

#         if self.best_score is None:
#             self.best_score = score
#             self.best_epoch = epoch + 1
#         elif score > self.best_score - self.delta:
#             self.counter += 1
#             if verbose:
#                 print("EarlyStopping counter: {} out of {}...".format(
#                     self.counter, self.patience))
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.best_epoch = epoch + 1
#             self.counter = 0
