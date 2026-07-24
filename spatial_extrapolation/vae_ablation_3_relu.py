#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import random

# =============================================================================
# 0. Data Preprocessing Configuration
# =============================================================================

DATA_PATH = '../../denormalized_camels_data_time.parquet'
TRAIN_DATES = ('1989-01-01', '2001-12-31')
TEST_DATES = ('2002-01-01', '2004-12-31')

METEO_FEATURES = ['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']
TARGET = 'QObs(mm/d)'
STATIC_FEATURES = [
    'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow', 'aridity',
    'high_prec_freq', 'high_prec_dur', 'low_prec_freq', 'low_prec_dur',
    'frac_forest', 'lai_max', 'lai_diff', 'gvf_max', 'gvf_diff',
    'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
    'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac',
    'clay_frac', 'elev_mean', 'slope_mean', 'area_gages2',
    'carbonate_rocks_frac', 'geol_permeability'
]


# =============================================================================
# 1. VAE Model Architecture
# =============================================================================

class BasinFlowVAE(nn.Module):
    """
    VAE for learning basin streamflow distributions.

    Architecture:
    - Encoder: basin features + flow sequence -> latent distribution params (mu, sigma)
    - Decoder: basin features + latent code -> flow distribution params
    """

    def __init__(self,
                 feature_dim: int = 37,
                 flow_dim: int = 365,
                 latent_dim: int = 32,
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()

        self.feature_dim = feature_dim
        self.flow_dim = flow_dim
        self.latent_dim = latent_dim

        # ========== Encoder ==========
        # Input: basin features + flow data
        encoder_input_dim = feature_dim + flow_dim

        encoder_layers = []
        in_dim = encoder_input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # ========== Decoder ==========
        # Input: basin features + latent code
        decoder_input_dim = feature_dim + latent_dim

        decoder_layers = []
        in_dim = decoder_input_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        self.decoder = nn.Sequential(*decoder_layers)

        # Output flow distribution parameters
        self.fc_flow_mu = nn.Linear(hidden_dims[0], flow_dim)
        self.fc_flow_logvar = nn.Linear(hidden_dims[0], flow_dim)

    def encode(self, features: torch.Tensor, flows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder: encode basin features and flow into latent distribution."""
        x = torch.cat([features, flows], dim=1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decoder: generate flow distribution from latent code."""
        x = torch.cat([z, features], dim=1)
        h = self.decoder(x)
        flow_mu = self.fc_flow_mu(h)
        flow_logvar = self.fc_flow_logvar(h)
        return flow_mu, flow_logvar

    def forward(self, features: torch.Tensor, flows: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode
        z_mu, z_logvar = self.encode(features, flows)
        z = self.reparameterize(z_mu, z_logvar)

        # Decode
        flow_mu, flow_logvar = self.decode(z, features)

        return {
            'flow_mu': flow_mu,
            'flow_logvar': flow_logvar,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'z': z
        }

    def generate(self, features: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """Generate flow samples for a new basin."""
        self.eval()
        with torch.no_grad():
            batch_size = features.shape[0]
            samples = []

            for _ in range(n_samples):
                # Sample latent variable from standard normal
                z = torch.randn(batch_size, self.latent_dim).to(features.device)

                # Decode to generate flow
                flow_mu, flow_logvar = self.decode(z, features)
                flow_std = torch.exp(0.5 * flow_logvar)

                # Sample flow values
                flow_sample = flow_mu + torch.randn_like(flow_mu) * flow_std
                samples.append(flow_sample.unsqueeze(1))

            return torch.cat(samples, dim=1)  # [batch, n_samples, flow_dim]


# =============================================================================
# 2. Conditional VAE Variant (Recommended)
# =============================================================================

class ConditionalBasinVAE(nn.Module):
    """
    Conditional VAE: better suited for missing data scenarios.
    Latent variable z captures shared patterns across basins,
    conditioned on basin features c.
    """

    def __init__(self,
                 feature_dim: int = 37,
                 latent_dim: int = 16,
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()

        # Encoder: encode flow statistics to latent space
        self.flow_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Flow mean and std
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.z_mu = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: conditioned on basin features
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),  # ReLU activation added
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Output flow mean and std
        )

    def encode(self, flow_stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode flow statistics y to latent distribution z."""
        h = self.flow_encoder(flow_stats)
        z_mu = self.z_mu(h)
        z_logvar = self.z_logvar(h)
        return z_mu, z_logvar

    def decode(self, z: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Decode: generate flow statistics y from latent code z and basin features x."""
        feature_h = self.feature_encoder(features)
        combined = torch.cat([z, feature_h], dim=1)
        flow_stats = self.decoder(combined)
        return flow_stats

    def forward(self, features: torch.Tensor, flow_stats: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        z_mu, z_logvar = self.encode(flow_stats)

        # Reparameterize
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + eps * std

        # Decode
        flow_pred = self.decode(z, features)

        return {
            'flow_pred': flow_pred,
            'z_mu': z_mu,
            'z_logvar': z_logvar
        }

    def predict_new_basin(self, features: torch.Tensor, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """Predict flow distribution for a new basin."""
        self.eval()
        with torch.no_grad():
            samples = []

            for _ in range(n_samples):
                # Sample z from prior z ~ N(0, I)
                z = torch.randn(features.shape[0], self.z_mu.out_features).to(features.device)

                # Decode to generate flow statistics
                flow_stats = self.decode(z, features)
                samples.append(flow_stats)

            samples = torch.stack(samples, dim=1)  # [batch, n_samples, 2]

            # Compute statistics
            mean_estimate = samples.mean(dim=1).cpu().numpy()
            std_estimate = samples.std(dim=1).cpu().numpy()
            percentiles = np.percentile(samples.cpu().numpy(), [5, 25, 50, 75, 95], axis=1)

            return {
                'mean': mean_estimate,
                'std': std_estimate,
                'percentiles': percentiles,
                'samples': samples.cpu().numpy()
            }


# =============================================================================
# 3. Loss Function
# =============================================================================

def vae_loss(outputs: Dict[str, torch.Tensor],
             targets: torch.Tensor,
             beta: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    VAE loss = reconstruction loss + beta * KL divergence.

    Args:
        beta: weight for KL term (beta-VAE)
    """

    # Reconstruction loss
    flow_mu = outputs['flow_mu'] if 'flow_mu' in outputs else outputs['flow_pred']
    recon_loss = F.mse_loss(flow_mu, targets, reduction='mean')

    # KL divergence
    z_mu = outputs['z_mu']
    z_logvar = outputs['z_logvar']
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()) / z_mu.shape[0]

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return {
        'total': total_loss,
        'recon': recon_loss,
        'kl': kl_loss
    }


# =============================================================================
# 4. Dataset
# =============================================================================

class BasinDataset(Dataset):
    """Basin dataset."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, basin_ids: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.basin_ids = basin_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.basin_ids[idx]


# =============================================================================
# 5. Training Function
# =============================================================================

def train_vae(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              lr: float = 1e-3,
              beta_schedule: str = 'linear',
              beta_value: float = 0.1,
              random_seed: int = 42) -> Dict[str, List[float]]:
    """Train the VAE model."""

    # Set random seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_loss': [], 'train_recon': [],
               'train_kl': [], 'val_recon': [], 'val_kl': []}

    for epoch in range(epochs):
        if beta_schedule == 'linear':
            beta = min(beta_value, (epoch / (epochs // 2)) * beta_value)
        else:
            beta = beta_value

        # Training
        model.train()
        train_losses = {'total': 0, 'recon': 0, 'kl': 0}

        for features, flow_stats, _ in train_loader:
            features = features.to(device)
            flow_stats = flow_stats.to(device)

            optimizer.zero_grad()
            outputs = model(features, flow_stats)
            losses = vae_loss(outputs, flow_stats, beta=beta)

            losses['total'].backward()
            optimizer.step()

            for key in train_losses:
                train_losses[key] += losses[key].item()

        # Validation
        model.eval()
        val_losses = {'total': 0, 'recon': 0, 'kl': 0}

        with torch.no_grad():
            for features, flow_stats, _ in val_loader:
                features = features.to(device)
                flow_stats = flow_stats.to(device)

                outputs = model(features, flow_stats)
                losses = vae_loss(outputs, flow_stats, beta=beta)

                for key in val_losses:
                    val_losses[key] += losses[key].item()

        # Record history
        n_train = len(train_loader)
        n_val = len(val_loader)

        history['train_loss'].append(train_losses['total'] / n_train)
        history['train_recon'].append(train_losses['recon'] / n_train)
        history['train_kl'].append(train_losses['kl'] / n_train)
        history['val_loss'].append(val_losses['total'] / n_val)
        history['val_recon'].append(val_losses['recon'] / n_val)
        history['val_kl'].append(val_losses['kl'] / n_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {history['train_loss'][-1]:.4f} "
                  f"(Recon: {history['train_recon'][-1]:.4f}, KL: {history['train_kl'][-1]:.4f})")
            print(f"  Val Loss: {history['val_loss'][-1]:.4f} "
                  f"(Recon: {history['val_recon'][-1]:.4f}, KL: {history['val_kl'][-1]:.4f})")

    return history


