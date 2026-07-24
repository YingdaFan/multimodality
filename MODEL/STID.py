import torch
from torch import nn
import numpy as np

class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual
        return hidden

class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID

    Adapted to work with river-dl preprocessing pipeline.
    Input shape from DataLoader: [batch_size, seq_len, n_features]
    """

    def __init__(self, input_dim, hidden_dim, adj_matrix, remap_matrix=None, recur_dropout=0, dropout=0.15, return_states=False, device='cuda',
                 seed=None):
        super().__init__()

        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        # Basic attributes
        self.num_nodes = adj_matrix.shape[0]
        self.input_len = 365
        self.input_dim = input_dim
        self.output_len = 365
        self.device = device

        # Model hyperparameters - improved from original
        # Use hidden_dim to control model capacity
        self.embed_dim = max(32, hidden_dim // 2)  # At least 32, scaled with hidden_dim
        self.node_dim = max(8, hidden_dim // 8)     # At least 8, scaled with hidden_dim
        self.num_layer = 3                          # Increased from 2 to 3

        # Temporal embedding dimensions
        self.temp_dim_doy = 8   # Day of year embedding
        self.temp_dim_dow = 4   # Day of week embedding

        # Feature flags
        # NOTE: Temporal features are disabled by default because we don't have access
        # to actual time information in the forward pass. To enable them properly,
        # you should add day_of_year and day_of_week as input features during preprocessing.
        self.if_day_of_year = False  # Disable (no reliable time info in forward)
        self.if_day_of_week = False  # Disable (no reliable time info in forward)
        self.if_spatial = True        # Enable spatial embeddings

        # Compensate for disabled temporal features by using larger embeddings
        if not self.if_day_of_year and not self.if_day_of_week:
            self.embed_dim = max(48, hidden_dim)  # Increase embedding capacity
            self.node_dim = max(16, hidden_dim // 4)  # Increase spatial capacity

        # Spatial embeddings (learned per-node embeddings)
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # Temporal embeddings
        if self.if_day_of_year:
            self.day_of_year_emb = nn.Parameter(torch.empty(366, self.temp_dim_doy))  # 366 days (leap year)
            nn.init.xavier_uniform_(self.day_of_year_emb)

        if self.if_day_of_week:
            self.day_of_week_emb = nn.Parameter(torch.empty(7, self.temp_dim_dow))  # 7 days of week
            nn.init.xavier_uniform_(self.day_of_week_emb)

        # Time series embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True
        )

        # Calculate total hidden dimension
        self.hidden_dim = (
            self.embed_dim +
            self.node_dim * int(self.if_spatial) +
            self.temp_dim_doy * int(self.if_day_of_year) +
            self.temp_dim_dow * int(self.if_day_of_week)
        )

        # Encoder (stack of MLP blocks with residual connections)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # Regression layer for final prediction
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True
        )

        # Store remap matrix if provided (for spatial remapping)
        if remap_matrix is not None:
            self.remap_matrix = torch.from_numpy(remap_matrix).float().to(device)

    def forward(self, history_data, init_states=None):
        """
        Feed forward of STID adapted for river-dl data format.

        Args:
            history_data: Input tensor with shape [batch_size, seq_len, n_features]
                         where batch_size = num_windows * num_basins
                         seq_len = 365 (time steps)
                         n_features = input_dim (e.g., 33 features)
            init_states: Dummy parameter for compatibility (not used)

        Returns:
            prediction: Output tensor with shape [batch_size, seq_len, 1]
        """
        # Input shape: [batch_size, seq_len, n_features]
        batch_size, seq_len, _ = history_data.shape

        # Reshape to [batch_size, seq_len, 1, n_features] to match STID's expected format
        # Then permute to [batch_size, 1, n_features, seq_len] for Conv2d operations
        input_data = history_data.unsqueeze(2)  # [batch_size, seq_len, 1, n_features]

        # Prepare input for time series embedding
        # Shape: [batch_size, 1, n_features, seq_len] -> [batch_size, n_features*seq_len, 1, 1]
        input_data = input_data.permute(0, 2, 3, 1)  # [batch_size, 1, n_features, seq_len]
        input_data = input_data.reshape(batch_size, 1, -1).unsqueeze(-1)  # [batch_size, 1, n_features*seq_len, 1]
        input_data = input_data.transpose(1, 2)  # [batch_size, n_features*seq_len, 1, 1]

        # Time series embedding
        time_series_emb = self.time_series_emb_layer(input_data)  # [batch_size, embed_dim, 1, 1]

        # Prepare embedding list
        embeddings = [time_series_emb]

        # Spatial embeddings (use a single representative node embedding)
        if self.if_spatial:
            # Use mean of all node embeddings as a global spatial feature
            # Or alternatively, use a learnable parameter for batch-level spatial encoding
            spatial_emb = self.node_emb.mean(dim=0, keepdim=True)  # [1, node_dim]
            spatial_emb = spatial_emb.unsqueeze(0).unsqueeze(-1)  # [1, node_dim, 1, 1]
            spatial_emb = spatial_emb.expand(batch_size, -1, 1, 1)  # [batch_size, node_dim, 1, 1]
            embeddings.append(spatial_emb)

        # Temporal embeddings (if enabled)
        # Note: Currently disabled by default. To enable, set if_day_of_year/if_day_of_week=True
        # and add temporal features to your input data during preprocessing
        if self.if_day_of_year:
            # Extract day_of_year from input (should be added as a feature during preprocessing)
            # For now, this is disabled
            pass

        if self.if_day_of_week:
            # Extract day_of_week from input (should be added as a feature during preprocessing)
            # For now, this is disabled
            pass

        # Concatenate all embeddings
        hidden = torch.cat(embeddings, dim=1)  # [batch_size, hidden_dim, 1, 1]

        # Encoding through MLP layers
        hidden = self.encoder(hidden)  # [batch_size, hidden_dim, 1, 1]

        # Regression to get predictions
        prediction = self.regression_layer(hidden)  # [batch_size, output_len, 1, 1]

        # Reshape output to [batch_size, seq_len, 1]
        prediction = prediction.squeeze(-1).squeeze(-1)  # [batch_size, output_len]
        prediction = prediction.unsqueeze(-1)  # [batch_size, output_len, 1]

        return prediction
