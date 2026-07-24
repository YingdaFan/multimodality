#!/usr/bin/env python3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# =============================================================================
# 0. Data Loading
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


def load_and_filter_data(data_path, start_date, end_date):
    """Load data and filter by date range."""
    print(f"Loading data: {data_path}")
    df = pd.read_parquet(data_path)
    df['Time'] = pd.to_datetime(df['Time'])
    mask = (df['Time'] >= start_date) & (df['Time'] <= end_date)
    df_filtered = df[mask].copy()
    print(f"  Filtered ({start_date} ~ {end_date}): {len(df_filtered)} rows")
    print(f"  Contains {df_filtered['basin_id'].nunique()} basins")
    return df_filtered


def compute_basin_statistics(df, meteo_features, target, static_features):
    """Compute statistics for each basin."""
    print("\nComputing basin statistics...")
    basins = df['basin_id'].unique()
    stats_list = []

    for basin in basins:
        basin_data = df[df['basin_id'] == basin]
        valid_mask = (basin_data[target].notna()) & (basin_data[target] >= 0)
        basin_data_valid = basin_data[valid_mask]

        if len(basin_data_valid) < 100:
            continue

        meteo_mean = basin_data_valid[meteo_features].mean().values
        meteo_std = basin_data_valid[meteo_features].std().values
        target_mean = basin_data_valid[target].mean()
        target_std = basin_data_valid[target].std()
        static = basin_data_valid[static_features].iloc[0].values

        stats = {
            'basin_id': basin,
            'meteo_mean': meteo_mean,
            'meteo_std': meteo_std,
            'static': static,
            'target_mean': target_mean,
            'target_std': target_std
        }
        stats_list.append(stats)

    print(f"  Successfully computed statistics for {len(stats_list)} basins")
    return stats_list


def build_feature_target_matrices(train_stats, test_stats):
    """Build feature matrix X and target matrix Y."""
    print("\nBuilding feature and target matrices...")

    def extract_matrices(stats_list):
        meteo_means = np.vstack([s['meteo_mean'] for s in stats_list])
        meteo_stds = np.vstack([s['meteo_std'] for s in stats_list])
        statics = np.vstack([s['static'] for s in stats_list])
        X = np.hstack([meteo_means, meteo_stds, statics])
        Y = np.column_stack([
            [s['target_mean'] for s in stats_list],
            [s['target_std'] for s in stats_list]
        ])
        basin_ids = np.array([s['basin_id'] for s in stats_list])
        return X, Y, basin_ids

    X_train, Y_train, train_basins = extract_matrices(train_stats)
    X_test, Y_test, test_basins = extract_matrices(test_stats)

    print(f"  Training set: X_train {X_train.shape}, Y_train {Y_train.shape}")
    print(f"  Test set: X_test {X_test.shape}, Y_test {Y_test.shape}")

    return X_train, Y_train, X_test, Y_test, train_basins, test_basins


def prepare_spatial_extrapolation_data():
    """Prepare spatial extrapolation data: 530 basins for training, 531st for testing."""
    # 1. Load data
    df_train = load_and_filter_data(DATA_PATH, TRAIN_DATES[0], TRAIN_DATES[1])

    # 2. Compute statistics
    train_stats = compute_basin_statistics(df_train, METEO_FEATURES, TARGET, STATIC_FEATURES)

    # 3. Build matrices (use train data only, then split manually)
    def extract_matrices(stats_list):
        meteo_means = np.vstack([s['meteo_mean'] for s in stats_list])
        meteo_stds = np.vstack([s['meteo_std'] for s in stats_list])
        statics = np.vstack([s['static'] for s in stats_list])
        X = np.hstack([meteo_means, meteo_stds, statics])
        Y = np.column_stack([
            [s['target_mean'] for s in stats_list],
            [s['target_std'] for s in stats_list]
        ])
        basin_ids = np.array([s['basin_id'] for s in stats_list])
        return X, Y, basin_ids

    X_all, Y_all, basin_ids = extract_matrices(train_stats)

    # 4. Spatial extrapolation setup: first 530 for training, 531st for testing
    print("\nSpatial Extrapolation Setup (Leave-One-Out)")

    X_train_530 = X_all[:530]
    Y_train_530 = Y_all[:530]
    train_basins = basin_ids[:530]

    X_new_basin = X_all[530:531]
    Y_new_basin = Y_all[530:531]
    new_basin_id = basin_ids[530]

    print(f"  Training set: first 530 basins")
    print(f"  Test set: Basin {new_basin_id} (531st, never seen during training)")
    print(f"  True values: mean={Y_new_basin[0, 0]:.3f} mm/day, std={Y_new_basin[0, 1]:.3f} mm/day")

    # 5. Normalize (based on 530 basins only)
    print("\nNormalizing features (based on 530 basins)...")
    X_mean = X_train_530.mean(axis=0)
    X_std = X_train_530.std(axis=0) + 1e-8
    Y_mean = Y_train_530.mean(axis=0)
    Y_std = Y_train_530.std(axis=0) + 1e-8

    X_train_normalized = (X_train_530 - X_mean) / X_std
    Y_train_normalized = (Y_train_530 - Y_mean) / Y_std
    X_new_normalized = (X_new_basin - X_mean) / X_std

    return {
        'X_train': X_train_normalized,
        'Y_train': Y_train_normalized,
        'X_test': X_new_normalized,
        'Y_test': Y_new_basin,
        'train_basins': train_basins,
        'test_basin_id': new_basin_id,
        'normalization': {'X_mean': X_mean, 'X_std': X_std, 'Y_mean': Y_mean, 'Y_std': Y_std}
    }


# =============================================================================
# 1. Point Estimation Methods (Baseline)
# =============================================================================

class PointEstimationBaseline:
    """Point estimation baseline methods (no uncertainty quantification)."""

    def __init__(self, method='xgboost'):
        self.method = method
        self.model = None

    def fit(self, X_train, Y_train):
        """Train model."""
        print(f"\nTraining {self.method.upper()} model...")

        if self.method == 'xgboost':
            self.model = MultiOutputRegressor(
                XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbosity=0, n_jobs=-1)
            )
        elif self.method == 'rf':
            self.model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=500, max_depth=None,
                                    min_samples_split=2, random_state=42, n_jobs=-1)
            )
        elif self.method == 'gbm':
            self.model = MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                        max_depth=5, random_state=42)
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.model.fit(X_train, Y_train)
        print(f"  {self.method.upper()} training complete")

    def predict(self, X_test):
        """Predict (point estimate only)."""
        pred = self.model.predict(X_test)
        return {
            'mean': pred,
            'std': np.zeros_like(pred),  # No uncertainty
            'samples': None
        }


# =============================================================================
# 2. MC Dropout (Neural Network + Dropout Uncertainty)
# =============================================================================

class MCDropoutNN(nn.Module):
    """Neural network with MC Dropout."""

    def __init__(self, input_dim, output_dim=2, hidden_dims=[128, 64], dropout_p=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MCDropoutPredictor:
    """MC Dropout uncertainty estimation."""

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout_p=0.3):
        self.model = MCDropoutNN(input_dim, output_dim=2,
                                hidden_dims=hidden_dims, dropout_p=dropout_p)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, X_train, Y_train, epochs=200, batch_size=32, lr=0.001):
        """Train model."""
        print("\nTraining MC Dropout model...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = F.mse_loss(pred, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

        print("  MC Dropout training complete")

    def predict(self, X_test, n_samples=1000):
        """Predict using MC Dropout sampling."""
        self.model.train()  # Keep dropout active

        X_tensor = torch.FloatTensor(X_test).to(self.device)
        samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X_tensor)
                samples.append(pred.cpu().numpy())

        samples = np.array(samples)  # [n_samples, batch, 2]

        return {
            'mean': samples.mean(axis=0),
            'std': samples.std(axis=0),
            'samples': samples.transpose(1, 0, 2)  # [batch, n_samples, 2]
        }


# =============================================================================
# 3. Deep Ensemble (Ensemble of Independent Models)
# =============================================================================

class SimpleNN(nn.Module):
    """Simple feedforward neural network."""

    def __init__(self, input_dim, output_dim=2, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepEnsemble:
    """Deep Ensemble uncertainty estimation."""

    def __init__(self, input_dim, n_models=10, hidden_dims=[128, 64]):
        self.n_models = n_models
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for _ in range(n_models):
            model = SimpleNN(input_dim, output_dim=2, hidden_dims=hidden_dims)
            model.to(self.device)
            self.models.append(model)

    def fit(self, X_train, Y_train, epochs=200, batch_size=32, lr=0.001):
        """Train all models."""
        print(f"\nTraining Deep Ensemble ({self.n_models} models)...")

        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(Y_train)
        )

        for i, model in enumerate(self.models):
            print(f"  Training model {i+1}/{self.n_models}...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Each model uses different random initialization and data order
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for epoch in range(epochs):
                for X_batch, Y_batch in loader:
                    X_batch = X_batch.to(self.device)
                    Y_batch = Y_batch.to(self.device)

                    optimizer.zero_grad()
                    pred = model(X_batch)
                    loss = F.mse_loss(pred, Y_batch)
                    loss.backward()
                    optimizer.step()

        print("  Deep Ensemble training complete")

    def predict(self, X_test):
        """Ensemble predictions from all models."""
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        predictions = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                pred = model(X_tensor)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)  # [n_models, batch, 2]

        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'samples': predictions.transpose(1, 0, 2)  # [batch, n_models, 2]
        }


# =============================================================================
# 4. Quantile Regression
# =============================================================================

class QuantileRegressionModel:
    """Quantile Regression for predicting quantiles."""

    def __init__(self, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
        self.quantiles = quantiles
        self.models = {}  # One model per quantile

    def fit(self, X_train, Y_train):
        """Train quantile models."""
        print(f"\nTraining Quantile Regression model...")

        # Model each output dimension (mean and std) separately
        for output_idx in range(Y_train.shape[1]):
            self.models[output_idx] = {}
            y = Y_train[:, output_idx]

            for q in self.quantiles:
                # Use GradientBoostingRegressor with quantile loss
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q,
                    n_estimators=500, learning_rate=0.05,
                    max_depth=5, random_state=42
                )
                model.fit(X_train, y)
                self.models[output_idx][q] = model

        print("  Quantile Regression training complete")

    def predict(self, X_test):
        """Predict quantiles."""
        predictions = {}

        for output_idx in range(2):  # Mean and std
            output_preds = []
            for q in self.quantiles:
                pred = self.models[output_idx][q].predict(X_test)
                output_preds.append(pred)
            predictions[output_idx] = np.array(output_preds)  # [n_quantiles, n_samples]

        # Compute mean and std estimates
        mean_estimate = np.array([predictions[0][2], predictions[1][2]]).T  # Median

        # Estimate std from IQR
        iqr_0 = predictions[0][3] - predictions[0][1]  # 75% - 25%
        iqr_1 = predictions[1][3] - predictions[1][1]
        std_estimate = np.array([iqr_0 / 1.349, iqr_1 / 1.349]).T  # IQR to std conversion

        return {
            'mean': mean_estimate,
            'std': std_estimate,
            'quantiles': predictions,
            'samples': None
        }


# =============================================================================
# 5. KNN-based Method (Weighted Average of Similar Basins)
# =============================================================================

class KNNBasinPredictor:
    """K-Nearest Neighbors basin prediction."""

    def __init__(self, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        self.X_train = None
        self.Y_train = None

    def fit(self, X_train, Y_train):
        """Train KNN model."""
        print(f"\nTraining KNN model (k={self.n_neighbors})...")
        self.X_train = X_train
        self.Y_train = Y_train
        self.knn.fit(X_train)
        print("  KNN training complete")

    def predict(self, X_test):
        """Predict using distance-weighted nearest neighbor average."""
        distances, indices = self.knn.kneighbors(X_test)

        # Distance-based weights (closer = higher weight)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted average
        neighbors_Y = self.Y_train[indices]  # [n_test, k, 2]
        mean_estimate = (neighbors_Y * weights[:, :, np.newaxis]).sum(axis=1)

        # Std: weighted standard deviation of neighbors
        std_estimate = np.sqrt((weights[:, :, np.newaxis] * (neighbors_Y - mean_estimate[:, np.newaxis, :])**2).sum(axis=1))

        return {
            'mean': mean_estimate,
            'std': std_estimate,
            'samples': neighbors_Y,  # Use neighbors as "samples"
            'neighbors': indices
        }


# =============================================================================
# 6. Evaluation and Visualization
# =============================================================================

def evaluate_predictions(predictions: Dict, Y_true: np.ndarray,
                        normalization: Dict, method_name: str):
    """Evaluate prediction results."""
    Y_mean = normalization['Y_mean']
    Y_std = normalization['Y_std']

    # Denormalize
    pred_mean_orig = predictions['mean'] * Y_std + Y_mean

    # Compute errors
    errors = {
        'mean_mae': np.abs(pred_mean_orig[0, 0] - Y_true[0, 0]),
        'mean_mape': np.abs(pred_mean_orig[0, 0] - Y_true[0, 0]) / Y_true[0, 0] * 100,
        'std_mae': np.abs(pred_mean_orig[0, 1] - Y_true[0, 1]),
        'std_mape': np.abs(pred_mean_orig[0, 1] - Y_true[0, 1]) / Y_true[0, 1] * 100,
    }


    print(f"Prediction results (original scale):")
    print(f"  Flow mean: {pred_mean_orig[0, 0]:.3f} mm/day (true: {Y_true[0, 0]:.3f})")
    print(f"  Flow std: {pred_mean_orig[0, 1]:.3f} mm/day (true: {Y_true[0, 1]:.3f})")

    if predictions['std'] is not None and predictions['std'].sum() > 0:
        pred_std_orig = predictions['std'] * Y_std
        print(f"\nUncertainty:")
        print(f"  Mean uncertainty: +/- {pred_std_orig[0, 0]:.3f} mm/day")
        print(f"  Std uncertainty: +/- {pred_std_orig[0, 1]:.3f} mm/day")

        # Check if true values fall within confidence interval
        mean_in_ci = (pred_mean_orig[0, 0] - 2*pred_std_orig[0, 0] <= Y_true[0, 0] <=
                      pred_mean_orig[0, 0] + 2*pred_std_orig[0, 0])
        std_in_ci = (pred_mean_orig[0, 1] - 2*pred_std_orig[0, 1] <= Y_true[0, 1] <=
                     pred_mean_orig[0, 1] + 2*pred_std_orig[0, 1])

        print(f"\n95% confidence interval check:")
        print(f"  Mean within CI: {'yes' if mean_in_ci else 'no'}")
        print(f"  Std within CI: {'yes' if std_in_ci else 'no'}")

    print(f"\nPrediction errors:")
    print(f"  Flow mean MAE: {errors['mean_mae']:.3f} mm/day")
    print(f"  Flow mean MAPE: {errors['mean_mape']:.2f}%")
    print(f"  Flow std MAE: {errors['std_mae']:.3f} mm/day")
    print(f"  Flow std MAPE: {errors['std_mape']:.2f}%")

    return errors, pred_mean_orig, predictions.get('std', None)


def visualize_comparison(results: Dict[str, Dict], Y_true: np.ndarray,
                        save_path='./spatial_extrapolation_comparison.png'):
    """Visualize comparison of all methods."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract results from all methods
    methods = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # 1. Flow Mean Prediction Comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(methods))
    means = [results[m]['pred_mean'][0, 0] for m in methods]
    stds = [results[m]['pred_std'][0, 0] if results[m]['pred_std'] is not None
            else 0 for m in methods]

    bars = ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(x_pos, means, yerr=stds, fmt='none', ecolor='black',
                capsize=5, capthick=2)
    ax.axhline(Y_true[0, 0], color='red', linestyle='--', linewidth=2,
               label=f'True: {Y_true[0, 0]:.3f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Flow Mean (mm/day)')
    ax.set_title('Flow Mean Prediction Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Flow Std Prediction Comparison
    ax = axes[0, 1]
    stds_pred = [results[m]['pred_mean'][0, 1] for m in methods]
    stds_unc = [results[m]['pred_std'][0, 1] if results[m]['pred_std'] is not None
                else 0 for m in methods]

    bars = ax.bar(x_pos, stds_pred, color=colors, alpha=0.7, edgecolor='black')
    ax.errorbar(x_pos, stds_pred, yerr=stds_unc, fmt='none', ecolor='black',
                capsize=5, capthick=2)
    ax.axhline(Y_true[0, 1], color='red', linestyle='--', linewidth=2,
               label=f'True: {Y_true[0, 1]:.3f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('Flow Std (mm/day)')
    ax.set_title('Flow Std Prediction Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Prediction Error Comparison (MAPE)
    ax = axes[1, 0]
    mean_mapes = [results[m]['errors']['mean_mape'] for m in methods]
    std_mapes = [results[m]['errors']['std_mape'] for m in methods]

    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width/2, mean_mapes, width, label='Mean MAPE',
           color='skyblue', edgecolor='black')
    ax.bar(x + width/2, std_mapes, width, label='Std MAPE',
           color='lightcoral', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Prediction Error Comparison (MAPE)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Uncertainty Quantification Comparison
    ax = axes[1, 1]
    has_uncertainty = [m for m in methods if results[m]['pred_std'] is not None
                      and results[m]['pred_std'].sum() > 0]

    if has_uncertainty:
        x_pos_unc = np.arange(len(has_uncertainty))
        mean_uncs = [results[m]['pred_std'][0, 0] for m in has_uncertainty]
        std_uncs = [results[m]['pred_std'][0, 1] for m in has_uncertainty]

        x = np.arange(len(has_uncertainty))
        width = 0.35
        ax.bar(x - width/2, mean_uncs, width, label='Mean Uncertainty',
               color='lightgreen', edgecolor='black')
        ax.bar(x + width/2, std_uncs, width, label='Std Uncertainty',
               color='plum', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(has_uncertainty, rotation=45, ha='right')
        ax.set_ylabel('Uncertainty (mm/day)')
        ax.set_title('Uncertainty Quantification Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Uncertainty Estimation', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Uncertainty Quantification Comparison')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: {save_path}")
    plt.show()


# =============================================================================
# 7. Main
# =============================================================================

def main():
    """Run all methods and compare."""

    # 1. Prepare data
    data = prepare_spatial_extrapolation_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    normalization = data['normalization']

    # 2. Initialize all methods
    methods = {
        'XGBoost': PointEstimationBaseline(method='xgboost'),
        'Random Forest': PointEstimationBaseline(method='rf'),
        'MC Dropout': MCDropoutPredictor(input_dim=X_train.shape[1], dropout_p=0.3),
        'Deep Ensemble (5)': DeepEnsemble(input_dim=X_train.shape[1], n_models=5),
        'Quantile Regression': QuantileRegressionModel(),
        'KNN (k=10)': KNNBasinPredictor(n_neighbors=10),
    }

    # 3. Train and evaluate all methods
    results = {}

    for name, model in methods.items():
        print(f"\nMethod: {name}")

        # Train
        if hasattr(model, 'fit'):
            if name in ['MC Dropout', 'Deep Ensemble (5)']:
                model.fit(X_train, Y_train, epochs=200)
            else:
                model.fit(X_train, Y_train)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        errors, pred_mean, pred_std = evaluate_predictions(
            predictions, Y_test, normalization, name
        )

        results[name] = {
            'predictions': predictions,
            'errors': errors,
            'pred_mean': pred_mean,
            'pred_std': pred_std if pred_std is not None else np.zeros_like(pred_mean)
        }

    # 4. Visualize comparison
    print(f"\nGenerating comparison visualization")
    visualize_comparison(results, Y_test,
                        save_path='./spatial_extrapolation_comparison.png')


    best_mean_method = min(results.keys(),
                          key=lambda k: results[k]['errors']['mean_mape'])
    best_std_method = min(results.keys(),
                         key=lambda k: results[k]['errors']['std_mape'])

    return results


if __name__ == "__main__":
    results = main()
