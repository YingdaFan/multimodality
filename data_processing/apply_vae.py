#!/usr/bin/env python3
"""
Call VAE model to predict y_mean and y_std for basins, and update prepped.npz
Avoid information leakage in spatial extrapolation experiments

Important: Use data directly from prepped.npz, no longer reading from CSV

Two sets of statistics design:
1. y_mean / y_std: For LSTM training normalization
   - Non-masked basins: true values
   - Masked basins: VAE predicted values (due to no observation data)

2. y_mean_vae / y_std_vae: For LSTM denormalization -> Stage 2 input
   - All basins: VAE predicted values (simulating real application scenario)
"""

import sys
import os
import numpy as np
import argparse
import random

# ============================================
# Fix all random seeds for experiment reproducibility
# ============================================
GLOBAL_SEED = 42

def set_global_seed(seed=GLOBAL_SEED):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch seeds set after import
    os.environ['PYTHONHASHSEED'] = str(seed)

# Add spatial_extrapolation path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../spatial_extrapolation'))

# Import VAE model components
from vae_ablation_7_res import (
    ConditionalBasinVAE,
    BasinDataset,
    train_vae,
    STATIC_FEATURES,
)

import torch
from torch.utils.data import DataLoader

# Set all random seeds (including PyTorch)
set_global_seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"[INFO] Global random seed set to {GLOBAL_SEED}")


def compute_basin_statistics_from_npz(x_trn, ids_trn, basin_names, x_vars, y_mean_arr, y_std_arr):
    """
    Compute statistics for each basin directly from prepped.npz data

    Parameters:
    -----------
    x_trn : np.ndarray
        Training features (n_samples, seq_len, n_features)
    ids_trn : np.ndarray
        Basin ID for each sample (n_samples, seq_len, 1)
    basin_names : np.ndarray
        All basin names
    x_vars : list
        List of feature names
    y_mean_arr : np.ndarray
        Y mean for each basin (n_basins,) - true values computed during preprocessing
    y_std_arr : np.ndarray
        Y std for each basin (n_basins,) - true values computed during preprocessing

    Returns:
    --------
    dict: basin_id -> {'x_mean': array, 'x_std': array, 'y_mean': float, 'y_std': float}
    """
    n_samples = x_trn.shape[0]
    n_features = x_trn.shape[2]

    # Create mapping from basin name to index
    basin_name_to_idx = {name: idx for idx, name in enumerate(basin_names)}

    # Create mapping from basin_id to sample indices
    basin_to_samples = {basin: [] for basin in basin_names}
    for i in range(n_samples):
        basin_id = ids_trn[i, 0, 0]
        if basin_id in basin_to_samples:
            basin_to_samples[basin_id].append(i)

    # Compute statistics for each basin
    basin_stats = {}
    for basin_id, sample_indices in basin_to_samples.items():
        if len(sample_indices) == 0:
            continue

        # Get all X data for this basin (X is not masked)
        x_basin = x_trn[sample_indices]  # (n_windows, seq_len, n_features)

        # Flatten time dimension
        x_flat = x_basin.reshape(-1, n_features)  # (n_windows * seq_len, n_features)

        # Compute X statistics
        x_mean = np.nanmean(x_flat, axis=0)
        x_std = np.nanstd(x_flat, axis=0)

        # Use true Y statistics from preprocessing (not from y_obs_trn, as it may be NaN)
        basin_idx = basin_name_to_idx[basin_id]
        y_mean = float(y_mean_arr[basin_idx])
        y_std = float(y_std_arr[basin_idx])

        basin_stats[basin_id] = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std
        }

    return basin_stats


def predict_single_basin(masked_basin_name, basin_names, basin_stats, all_masked_basins):
    """
    Train VAE for a single masked basin and predict its y statistics

    Parameters:
    -----------
    masked_basin_name : str
        Name of the basin to predict
    basin_names : np.ndarray
        All basin names
    basin_stats : dict
        Statistics for all basins
    all_masked_basins : list
        List of all masked basin names

    Returns:
    --------
    tuple: (predicted_mean, predicted_std, true_mean, true_std)
    """
    print(f"\nIndependently training VAE to predict basin: {masked_basin_name}")

    # Get true statistics for masked basin (for comparison)
    if masked_basin_name not in basin_stats:
        raise ValueError(f"Basin {masked_basin_name} not found in statistics")

    true_stats = basin_stats[masked_basin_name]
    true_mean = true_stats['y_mean']
    true_std = true_stats['y_std']

    # Collect data from unmasked basins for training
    X_unmask = []
    Y_unmask = []
    unmask_basin_names = []

    for basin_name in basin_names:
        if basin_name in all_masked_basins:
            continue
        if basin_name not in basin_stats:
            continue

        stats = basin_stats[basin_name]
        # Skip basins with no observation data (y_mean/y_std are NaN)
        if np.isnan(stats['y_mean']) or np.isnan(stats['y_std']):
            continue

        # Features: concatenate [x_mean, x_std]
        x_features = np.concatenate([stats['x_mean'], stats['x_std']])
        y_target = np.array([stats['y_mean'], stats['y_std']])

        X_unmask.append(x_features)
        Y_unmask.append(y_target)
        unmask_basin_names.append(basin_name)

    X_unmask = np.array(X_unmask)
    Y_unmask = np.array(Y_unmask)
    unmask_basin_names = np.array(unmask_basin_names)

    print(f"  Training data: {len(unmask_basin_names)} unmasked basins")
    print(f"  Feature dimension: {X_unmask.shape[1]}")

    # Get features for masked basin
    masked_stats = basin_stats[masked_basin_name]
    X_masked = np.concatenate([masked_stats['x_mean'], masked_stats['x_std']]).reshape(1, -1)

    # Standardize (based on unmasked basins only)
    X_mean = X_unmask.mean(axis=0)
    X_std_norm = X_unmask.std(axis=0) + 1e-8
    Y_mean_norm = Y_unmask.mean(axis=0)
    Y_std_norm = Y_unmask.std(axis=0) + 1e-8

    X_unmask_normed = (X_unmask - X_mean) / X_std_norm
    Y_unmask_normed = (Y_unmask - Y_mean_norm) / Y_std_norm
    X_masked_normed = (X_masked - X_mean) / X_std_norm

    # Train VAE
    print(f"  Training VAE model...")
    train_dataset = BasinDataset(X_unmask_normed, Y_unmask_normed, unmask_basin_names)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Hyperparameters (based on previous tuning results)
    model = ConditionalBasinVAE(
        feature_dim=X_unmask.shape[1],
        latent_dim=16,
        hidden_dim=128,
        dropout=0.0
    )
    history = train_vae(
        model, train_loader, train_loader,
        epochs=100, lr=0.001,
        beta_schedule='constant', beta_value=0.1
    )

    # Predict
    print(f"  Using VAE to predict basin {masked_basin_name} y statistics...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_features = torch.FloatTensor(X_masked_normed).to(device)
    predictions_norm = model.predict_new_basin(masked_features, n_samples=1000)

    # Denormalization
    predictions = predictions_norm['mean'] * Y_std_norm + Y_mean_norm

    pred_mean = predictions[0, 0]
    pred_std = predictions[0, 1]

    print(f"\n  Prediction results:")
    print(f"    True Mean: {true_mean:.3f}, Pred Mean: {pred_mean:.3f}, Error: {abs(pred_mean-true_mean)/abs(true_mean)*100:.2f}%")
    print(f"    True Std:  {true_std:.3f}, Pred Std:  {pred_std:.3f}, Error: {abs(pred_std-true_std)/abs(true_std)*100:.2f}%")

    return pred_mean, pred_std, true_mean, true_std


def predict_all_basins_with_vae(basin_names, basin_stats, masked_basin_names):
    """
    Train a VAE model (using non-masked basins), then predict for all basins.
    Used to generate y_mean_vae / y_std_vae (all basins predicted by VAE).

    Parameters:
    -----------
    basin_names : np.ndarray
        All basin names
    basin_stats : dict
        Statistics for all basins
    masked_basin_names : list
        List of masked basin names

    Returns:
    --------
    dict: basin_id -> {'pred_mean': float, 'pred_std': float, 'true_mean': float, 'true_std': float}
    """
    print("\n" + "=" * 60)
    print("Train VAE model and predict for all basins (for y_mean_vae / y_std_vae)")
    print("=" * 60)

    # Collect data from unmasked basins for training
    X_unmask = []
    Y_unmask = []
    unmask_basin_names = []

    for basin_name in basin_names:
        if basin_name in masked_basin_names:
            continue
        if basin_name not in basin_stats:
            continue

        stats = basin_stats[basin_name]
        # Skip basins with no observation data (y_mean/y_std are NaN)
        if np.isnan(stats['y_mean']) or np.isnan(stats['y_std']):
            print(f"  Skipping no-data basin {basin_name} from VAE training (y_mean/y_std=NaN)")
            continue

        x_features = np.concatenate([stats['x_mean'], stats['x_std']])
        y_target = np.array([stats['y_mean'], stats['y_std']])

        X_unmask.append(x_features)
        Y_unmask.append(y_target)
        unmask_basin_names.append(basin_name)

    X_unmask = np.array(X_unmask)
    Y_unmask = np.array(Y_unmask)
    unmask_basin_names = np.array(unmask_basin_names)

    print(f"Training data: {len(unmask_basin_names)} unmasked basins (after filtering no-data)")
    print(f"Feature dimension: {X_unmask.shape[1]}")

    # Standardize (based on unmasked basins)
    X_mean = X_unmask.mean(axis=0)
    X_std_norm = X_unmask.std(axis=0) + 1e-8
    Y_mean_norm = Y_unmask.mean(axis=0)
    Y_std_norm = Y_unmask.std(axis=0) + 1e-8

    X_unmask_normed = (X_unmask - X_mean) / X_std_norm
    Y_unmask_normed = (Y_unmask - Y_mean_norm) / Y_std_norm

    # Train VAE
    print("Training VAE model...")
    train_dataset = BasinDataset(X_unmask_normed, Y_unmask_normed, unmask_basin_names)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Get hyperparameters from global variables (if available), otherwise use defaults
    hp = globals().get('VAE_HYPERPARAMS', {})
    model = ConditionalBasinVAE(
        feature_dim=X_unmask.shape[1],
        latent_dim=hp.get('latent_dim', 16),
        hidden_dim=hp.get('hidden_dim', 128),
        dropout=hp.get('dropout', 0.0)
    )
    history = train_vae(
        model, train_loader, train_loader,
        epochs=hp.get('epochs', 100), lr=hp.get('lr', 0.001),
        beta_schedule='constant', beta_value=hp.get('beta_value', 0.1)
    )

    # Predict for all basins
    print("\nPredicting all basins with VAE...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    all_predictions = {}
    for basin_name in basin_names:
        if basin_name not in basin_stats:
            continue

        stats = basin_stats[basin_name]
        true_mean = stats['y_mean']
        true_std = stats['y_std']

        # Prepare features
        x_features = np.concatenate([stats['x_mean'], stats['x_std']]).reshape(1, -1)
        x_normed = (x_features - X_mean) / X_std_norm

        # VAE prediction
        x_tensor = torch.FloatTensor(x_normed).to(device)
        predictions_norm = model.predict_new_basin(x_tensor, n_samples=1000)

        # Denormalization
        predictions = predictions_norm['mean'] * Y_std_norm + Y_mean_norm
        pred_mean = float(predictions[0, 0])
        pred_std = float(predictions[0, 1])

        all_predictions[basin_name] = {
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'true_mean': true_mean,
            'true_std': true_std
        }

    # Print summary
    print("\nVAE prediction results summary (all basins):")
    print("-" * 80)
    total_mean_err = 0
    total_std_err = 0
    count = 0
    for basin_name in basin_names:
        if basin_name not in all_predictions:
            continue
        pred = all_predictions[basin_name]
        true_mean = pred['true_mean']
        pred_mean = pred['pred_mean']
        true_std = pred['true_std']
        pred_std = pred['pred_std']

        # Skip basins with no observation data (NaN true statistics)
        if np.isnan(true_mean) or np.isnan(true_std):
            continue

        mean_err = abs(pred_mean - true_mean) / abs(true_mean) * 100 if true_mean != 0 else 0
        std_err = abs(pred_std - true_std) / abs(true_std) * 100 if true_std != 0 else 0
        total_mean_err += mean_err
        total_std_err += std_err
        count += 1

    avg_mean_err = total_mean_err / count if count > 0 else 0
    avg_std_err = total_std_err / count if count > 0 else 0
    print(f"Average Mean Error: {avg_mean_err:.2f}%")
    print(f"Average Std Error: {avg_std_err:.2f}%")
    print(f"Number of predicted basins: {count}")

    return all_predictions


def main(masked_basin_names, script_dir):
    """
    Main function:
    1. Independently train VAE for each masked basin and update y_mean/y_std
    2. Train a VAE to predict all basins, generating y_mean_vae/y_std_vae

    Parameters:
    -----------
    masked_basin_names : list of str
        List of masked basin names
    script_dir : str
        Script directory (for saving vae_statistics_log.csv)
    """
    # File path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npz_file = os.path.join(current_dir, 'data', 'prepped.npz')

    print(f"\nLoading npz file: {npz_file}")

    # 1. Load npz data
    data = np.load(npz_file, allow_pickle=True)
    basin_names = data['basin_names']
    x_trn = data['x_trn']
    y_obs_trn = data['y_obs_trn']
    ids_trn = data['ids_trn']
    x_vars = list(data['x_vars'])

    print(f"Number of basins: {len(basin_names)}")
    print(f"x_trn shape: {x_trn.shape}")
    print(f"y_obs_trn shape: {y_obs_trn.shape}")
    print(f"Number of features: {len(x_vars)}")

    # Find indices for masked basins
    masked_indices = []
    for basin_name in masked_basin_names:
        idx = np.where(basin_names == basin_name)[0]
        if len(idx) > 0:
            masked_indices.append(idx[0])
        else:
            print(f"Warning: Basin '{basin_name}' not found in data")

    if len(masked_indices) == 0:
        print("No valid masked basins found, VAE will train on all basins")
    else:
        print(f"Found {len(masked_indices)} masked basin(s) (indices: {masked_indices})")

    # 2. Compute statistics for each basin directly from npz data
    print("\nComputing statistics for each basin from prepped.npz...")
    basin_stats = compute_basin_statistics_from_npz(
        x_trn, ids_trn, basin_names, x_vars, data['y_mean'], data['y_std']
    )
    print(f"Successfully computed statistics for {len(basin_stats)}  basin(s)")

    # ============================================================
    # New logic: train 1 VAE, predict all basins (unified model)
    # ============================================================
    # Training set: exclude all masked basins
    # Prediction: all basins (including masked and non-masked)
    # Advantage: masked basins have consistent values in y_mean/y_std and y_mean_vae/y_std_vae
    all_vae_predictions = predict_all_basins_with_vae(
        basin_names, basin_stats, masked_basin_names
    )

    # 3. Print prediction results for masked basins
    print("\n" + "=" * 80)
    print("VAE prediction results summary (masked basins):")
    print("=" * 80)
    print(f"{'Basin':<15} {'True Mean':<12} {'Pred Mean':<12} {'True Std':<12} {'Pred Std':<12} {'Mean Err%':<12} {'Std Err%':<12}")
    print("-" * 80)

    for basin_name in masked_basin_names:
        if basin_name not in all_vae_predictions:
            continue
        pred = all_vae_predictions[basin_name]
        true_mean = pred['true_mean']
        pred_mean = pred['pred_mean']
        true_std = pred['true_std']
        pred_std = pred['pred_std']

        mean_err = abs(pred_mean - true_mean) / abs(true_mean) * 100 if true_mean != 0 else 0
        std_err = abs(pred_std - true_std) / abs(true_std) * 100 if true_std != 0 else 0

        print(f"{basin_name:<15} {true_mean:<12.3f} {pred_mean:<12.3f} "
              f"{true_std:<12.3f} {pred_std:<12.3f} {mean_err:<12.2f} {std_err:<12.2f}")

    # 4. Update npz file (using the same VAE prediction results uniformly)
    print("\n" + "=" * 60)
    print("Update NPZ file (using the same VAE prediction results uniformly)")
    print("=" * 60)
    data_dict = {key: data[key] for key in data.files}

    # 4.1 Update y_mean/y_std (VAE predictions for masked basins, true values for non-masked)
    print("\n[1] y_mean/y_std: VAE predictions for masked basins")
    y_mean_new = data_dict['y_mean'].copy()
    y_std_new = data_dict['y_std'].copy()

    for basin_idx, basin_name in zip(masked_indices, masked_basin_names):
        if basin_name not in all_vae_predictions:
            continue

        old_mean = y_mean_new[basin_idx]
        old_std = y_std_new[basin_idx]
        new_mean = all_vae_predictions[basin_name]['pred_mean']
        new_std = all_vae_predictions[basin_name]['pred_std']

        y_mean_new[basin_idx] = new_mean
        y_std_new[basin_idx] = new_std

        print(f"  Basin {basin_name}:")
        print(f"    y_mean: {old_mean:.4f} -> {new_mean:.4f}")
        print(f"    y_std:  {old_std:.4f} -> {new_std:.4f}")

    data_dict['y_mean'] = y_mean_new
    data_dict['y_std'] = y_std_new

    # 4.2 Generate y_mean_vae/y_std_vae (VAE predictions for all basins)
    n_basins = len(basin_names)
    y_mean_vae = np.zeros(n_basins)
    y_std_vae = np.zeros(n_basins)

    for i, basin_name in enumerate(basin_names):
        if basin_name in all_vae_predictions:
            y_mean_vae[i] = all_vae_predictions[basin_name]['pred_mean']
            y_std_vae[i] = all_vae_predictions[basin_name]['pred_std']
        else:
            # Fallback: use true values (should not happen)
            y_mean_vae[i] = data_dict['y_mean'][i]
            y_std_vae[i] = data_dict['y_std'][i]
            print(f"  Warning: Basin {basin_name} VAE prediction failed, using true values")

    data_dict['y_mean_vae'] = y_mean_vae
    data_dict['y_std_vae'] = y_std_vae

    print(f"y_mean_vae shape: {y_mean_vae.shape}")
    print(f"y_std_vae shape: {y_std_vae.shape}")

    # Save updated data
    np.savez_compressed(npz_file, **data_dict)

    # 6. Write VAE prediction results to CSV file
    import pandas as pd
    from datetime import datetime

    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    vae_stats_log = os.path.join(output_dir, 'vae_statistics_log.csv')

    # Check if file exists
    if os.path.exists(vae_stats_log):
        df_vae = pd.read_csv(vae_stats_log)
    else:
        df_vae = pd.DataFrame(columns=['timestamp', 'basin', 'y_mean_true', 'y_mean_vae',
                                        'y_std_true', 'y_std_vae', 'mean_error_pct', 'std_error_pct'])

    # Add record for each basin
    for basin_name in masked_basin_names:
        if basin_name not in all_vae_predictions:
            continue

        pred = all_vae_predictions[basin_name]
        true_mean = pred['true_mean']
        pred_mean = pred['pred_mean']
        true_std = pred['true_std']
        pred_std = pred['pred_std']

        mean_err_pct = abs(pred_mean - true_mean) / abs(true_mean) * 100 if true_mean != 0 else 0
        std_err_pct = abs(pred_std - true_std) / abs(true_std) * 100 if true_std != 0 else 0

        # Delete old record for this basin
        df_vae = df_vae[df_vae['basin'] != basin_name]

        # Add new record
        new_row = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basin': basin_name,
            'y_mean_true': f'{true_mean:.6f}',
            'y_mean_vae': f'{pred_mean:.6f}',
            'y_std_true': f'{true_std:.6f}',
            'y_std_vae': f'{pred_std:.6f}',
            'mean_error_pct': f'{mean_err_pct:.2f}',
            'std_error_pct': f'{std_err_pct:.2f}'
        }])
        df_vae = pd.concat([df_vae, new_row], ignore_index=True)

    # Write to CSV
    df_vae.to_csv(vae_stats_log, index=False)
    print(f"\nVAE statistics saved to: {vae_stats_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply VAE correction for masked basins')
    parser.add_argument('basins', nargs='+', help='Basin IDs to mask')
    parser.add_argument('--script_dir', type=str, required=True,
                        help='Directory where vae_statistics_log.csv will be saved')
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta_value', type=float, default=0.1)
    args = parser.parse_args()

    masked_basins = args.basins
    print(f"Received masked basins: {masked_basins}")

    globals()['VAE_HYPERPARAMS'] = vars(args)

    # Run main function
    main(masked_basins, args.script_dir)


# ============================================================
# Legacy logic backup (independently train VAE for each masked basin)
#!/usr/bin/env python3
"""
Call VAE model to predict y_mean and y_std for masked basins, and update prepped.npz
Avoid information leakage in spatial extrapolation experiments

Important: Use data directly from prepped.npz, no longer reading from CSV
"""

'''
import sys
import os
import numpy as np
import argparse

# Add spatial_extrapolation path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../spatial_extrapolation'))

# Import VAE model components
from vae_ablation_7_res import (
    ConditionalBasinVAE,
    BasinDataset,
    train_vae,
    STATIC_FEATURES,
)

import torch
from torch.utils.data import DataLoader


def compute_basin_statistics_from_npz(x_trn, ids_trn, basin_names, x_vars, y_mean_arr, y_std_arr):
    """
    Compute statistics for each basin directly from prepped.npz data

    Parameters:
    -----------
    x_trn : np.ndarray
        Training features (n_samples, seq_len, n_features)
    ids_trn : np.ndarray
        Basin ID for each sample (n_samples, seq_len, 1)
    basin_names : np.ndarray
        All basin names
    x_vars : list
        List of feature names
    y_mean_arr : np.ndarray
        Y mean for each basin (n_basins,) - true values computed during preprocessing
    y_std_arr : np.ndarray
        Y std for each basin (n_basins,) - true values computed during preprocessing

    Returns:
    --------
    dict: basin_id -> {'x_mean': array, 'x_std': array, 'y_mean': float, 'y_std': float}
    """
    n_samples = x_trn.shape[0]
    n_features = x_trn.shape[2]

    # Create mapping from basin name to index
    basin_name_to_idx = {name: idx for idx, name in enumerate(basin_names)}

    # Create mapping from basin_id to sample indices
    basin_to_samples = {basin: [] for basin in basin_names}
    for i in range(n_samples):
        basin_id = ids_trn[i, 0, 0]
        if basin_id in basin_to_samples:
            basin_to_samples[basin_id].append(i)

    # Compute statistics for each basin
    basin_stats = {}
    for basin_id, sample_indices in basin_to_samples.items():
        if len(sample_indices) == 0:
            continue

        # Get all X data for this basin (X is not masked)
        x_basin = x_trn[sample_indices]  # (n_windows, seq_len, n_features)

        # Flatten time dimension
        x_flat = x_basin.reshape(-1, n_features)  # (n_windows * seq_len, n_features)

        # Compute X statistics
        x_mean = np.nanmean(x_flat, axis=0)
        x_std = np.nanstd(x_flat, axis=0)

        # Use true Y statistics from preprocessing (not from y_obs_trn, as it may be NaN)
        basin_idx = basin_name_to_idx[basin_id]
        y_mean = float(y_mean_arr[basin_idx])
        y_std = float(y_std_arr[basin_idx])

        basin_stats[basin_id] = {
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std
        }

    return basin_stats


def predict_single_basin(masked_basin_name, basin_names, basin_stats, all_masked_basins):
    """
    Train VAE for a single masked basin and predict its y statistics

    Parameters:
    -----------
    masked_basin_name : str
        Name of the basin to predict
    basin_names : np.ndarray
        All basin names
    basin_stats : dict
        Statistics for all basins
    all_masked_basins : list
        List of all masked basin names

    Returns:
    --------
    tuple: (predicted_mean, predicted_std, true_mean, true_std)
    """
    print(f"\nIndependently training VAE to predict basin: {masked_basin_name}")

    # Get true statistics for masked basin (for comparison)
    if masked_basin_name not in basin_stats:
        raise ValueError(f"Basin {masked_basin_name} not found in statistics")

    true_stats = basin_stats[masked_basin_name]
    true_mean = true_stats['y_mean']
    true_std = true_stats['y_std']

    # Collect data from unmasked basins for training
    X_unmask = []
    Y_unmask = []
    unmask_basin_names = []

    for basin_name in basin_names:
        if basin_name in all_masked_basins:
            continue
        if basin_name not in basin_stats:
            continue

        stats = basin_stats[basin_name]
        # Skip basins with no observation data (y_mean/y_std are NaN)
        if np.isnan(stats['y_mean']) or np.isnan(stats['y_std']):
            continue

        # Features: concatenate [x_mean, x_std]
        x_features = np.concatenate([stats['x_mean'], stats['x_std']])
        y_target = np.array([stats['y_mean'], stats['y_std']])

        X_unmask.append(x_features)
        Y_unmask.append(y_target)
        unmask_basin_names.append(basin_name)

    X_unmask = np.array(X_unmask)
    Y_unmask = np.array(Y_unmask)
    unmask_basin_names = np.array(unmask_basin_names)

    print(f"  Training data: {len(unmask_basin_names)} unmasked basins")
    print(f"  Feature dimension: {X_unmask.shape[1]}")

    # Get features for masked basin
    masked_stats = basin_stats[masked_basin_name]
    X_masked = np.concatenate([masked_stats['x_mean'], masked_stats['x_std']]).reshape(1, -1)

    # Standardize (based on unmasked basins only)
    X_mean = X_unmask.mean(axis=0)
    X_std_norm = X_unmask.std(axis=0) + 1e-8
    Y_mean_norm = Y_unmask.mean(axis=0)
    Y_std_norm = Y_unmask.std(axis=0) + 1e-8

    X_unmask_normed = (X_unmask - X_mean) / X_std_norm
    Y_unmask_normed = (Y_unmask - Y_mean_norm) / Y_std_norm
    X_masked_normed = (X_masked - X_mean) / X_std_norm

    # Train VAE
    print(f"  Training VAE model...")
    train_dataset = BasinDataset(X_unmask_normed, Y_unmask_normed, unmask_basin_names)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Hyperparameters (based on previous tuning results)
    model = ConditionalBasinVAE(
        feature_dim=X_unmask.shape[1],
        latent_dim=16,
        hidden_dim=128,
        dropout=0.0
    )
    history = train_vae(
        model, train_loader, train_loader,
        epochs=100, lr=0.001,
        beta_schedule='constant', beta_value=0.1
    )

    # Predict
    print(f"  Using VAE to predict basin {masked_basin_name} y statistics...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    masked_features = torch.FloatTensor(X_masked_normed).to(device)
    predictions_norm = model.predict_new_basin(masked_features, n_samples=1000)

    # Denormalization
    predictions = predictions_norm['mean'] * Y_std_norm + Y_mean_norm

    pred_mean = predictions[0, 0]
    pred_std = predictions[0, 1]

    print(f"\n  Prediction results:")
    print(f"    True Mean: {true_mean:.3f}, Pred Mean: {pred_mean:.3f}, Error: {abs(pred_mean-true_mean)/abs(true_mean)*100:.2f}%")
    print(f"    True Std:  {true_std:.3f}, Pred Std:  {pred_std:.3f}, Error: {abs(pred_std-true_std)/abs(true_std)*100:.2f}%")

    return pred_mean, pred_std, true_mean, true_std


def main(masked_basin_names, script_dir):
    """
    Main function: independently train VAE for each masked basin and update its statistics

    Parameters:
    -----------
    masked_basin_names : list of str
        List of masked basin names
    script_dir : str
        Script directory (for saving vae_statistics_log.csv)
    """
    print("=" * 50)
    print("Using VAE to predict y_mean and y_std for masked basins")
    print("Strategy: use prepped.npz data directly, independently train VAE model for each masked basin")
    print(f"Masked basins: {masked_basin_names}")
    print("=" * 50)

    # File path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    npz_file = os.path.join(current_dir, 'data', 'prepped.npz')

    print(f"\nLoading npz file: {npz_file}")

    # 1. Load npz data
    data = np.load(npz_file, allow_pickle=True)
    basin_names = data['basin_names']
    x_trn = data['x_trn']
    y_obs_trn = data['y_obs_trn']
    ids_trn = data['ids_trn']
    x_vars = list(data['x_vars'])

    print(f"Number of basins: {len(basin_names)}")
    print(f"x_trn shape: {x_trn.shape}")
    print(f"y_obs_trn shape: {y_obs_trn.shape}")
    print(f"Number of features: {len(x_vars)}")

    # Find indices for masked basins
    masked_indices = []
    for basin_name in masked_basin_names:
        idx = np.where(basin_names == basin_name)[0]
        if len(idx) > 0:
            masked_indices.append(idx[0])
        else:
            print(f"Warning: Basin '{basin_name}' not found in data")

    if len(masked_indices) == 0:
        print("No valid masked basins, exiting")
        return

    print(f"Found {len(masked_indices)} masked basin(s) (indices: {masked_indices})")

    # 2. Compute statistics for each basin directly from npz data
    print("\nComputing statistics for each basin from prepped.npz...")
    basin_stats = compute_basin_statistics_from_npz(
        x_trn, ids_trn, basin_names, x_vars, data['y_mean'], data['y_std']
    )
    print(f"Successfully computed statistics for {len(basin_stats)}  basin(s)")

    # 3. Independently train VAE for each masked basin and predict
    print(f"\nWill independently train VAE models for {len(masked_basin_names)}  basin(s)")

    predictions = {}
    for basin_name in masked_basin_names:
        try:
            pred_mean, pred_std, true_mean, true_std = predict_single_basin(
                basin_name, basin_names, basin_stats, masked_basin_names
            )
            predictions[basin_name] = {
                'pred_mean': pred_mean,
                'pred_std': pred_std,
                'true_mean': true_mean,
                'true_std': true_std
            }
        except Exception as e:
            print(f"Warning: predicting basin {basin_name}  failed: {e}")
            continue

    # 4. Print summary results
    print("\n" + "=" * 80)
    print("VAE prediction results summary:")
    print("=" * 80)
    print(f"{'Basin':<15} {'True Mean':<12} {'Pred Mean':<12} {'True Std':<12} {'Pred Std':<12} {'Mean Err%':<12} {'Std Err%':<12}")
    print("-" * 80)

    for basin_name in masked_basin_names:
        if basin_name not in predictions:
            continue
        pred = predictions[basin_name]
        true_mean = pred['true_mean']
        pred_mean = pred['pred_mean']
        true_std = pred['true_std']
        pred_std = pred['pred_std']

        mean_err = abs(pred_mean - true_mean) / abs(true_mean) * 100 if true_mean != 0 else 0
        std_err = abs(pred_std - true_std) / abs(true_std) * 100 if true_std != 0 else 0

        print(f"{basin_name:<15} {true_mean:<12.3f} {pred_mean:<12.3f} "
              f"{true_std:<12.3f} {pred_std:<12.3f} {mean_err:<12.2f} {std_err:<12.2f}")

    # 5. Update npz file
    print("\nUpdating y_mean and y_std in npz file...")
    data_dict = {key: data[key] for key in data.files}

    y_mean_new = data_dict['y_mean'].copy()
    y_std_new = data_dict['y_std'].copy()

    for basin_idx, basin_name in zip(masked_indices, masked_basin_names):
        if basin_name not in predictions:
            continue

        old_mean = y_mean_new[basin_idx]
        old_std = y_std_new[basin_idx]
        new_mean = predictions[basin_name]['pred_mean']
        new_std = predictions[basin_name]['pred_std']

        y_mean_new[basin_idx] = new_mean
        y_std_new[basin_idx] = new_std

        print(f"  Basin {basin_name}:")
        print(f"    y_mean: {old_mean:.4f} -> {new_mean:.4f}")
        print(f"    y_std:  {old_std:.4f} -> {new_std:.4f}")

    data_dict['y_mean'] = y_mean_new
    data_dict['y_std'] = y_std_new

    # Save updated data
    print(f"\nSaving updated data to: {npz_file}")
    np.savez_compressed(npz_file, **data_dict)

    print("\nDone! Successfully updated y_mean and y_std for masked basins")

    # 6. Write VAE prediction results to CSV file
    import pandas as pd
    from datetime import datetime

    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    vae_stats_log = os.path.join(output_dir, 'vae_statistics_log.csv')

    # Check if file exists
    if os.path.exists(vae_stats_log):
        df_vae = pd.read_csv(vae_stats_log)
    else:
        df_vae = pd.DataFrame(columns=['timestamp', 'basin', 'y_mean_true', 'y_mean_vae',
                                        'y_std_true', 'y_std_vae', 'mean_error_pct', 'std_error_pct'])

    # Add record for each basin
    for basin_name in masked_basin_names:
        if basin_name not in predictions:
            continue

        pred = predictions[basin_name]
        true_mean = pred['true_mean']
        pred_mean = pred['pred_mean']
        true_std = pred['true_std']
        pred_std = pred['pred_std']

        mean_err_pct = abs(pred_mean - true_mean) / abs(true_mean) * 100 if true_mean != 0 else 0
        std_err_pct = abs(pred_std - true_std) / abs(true_std) * 100 if true_std != 0 else 0

        # Delete old record for this basin
        df_vae = df_vae[df_vae['basin'] != basin_name]

        # Add new record
        new_row = pd.DataFrame([{
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basin': basin_name,
            'y_mean_true': f'{true_mean:.6f}',
            'y_mean_vae': f'{pred_mean:.6f}',
            'y_std_true': f'{true_std:.6f}',
            'y_std_vae': f'{pred_std:.6f}',
            'mean_error_pct': f'{mean_err_pct:.2f}',
            'std_error_pct': f'{std_err_pct:.2f}'
        }])
        df_vae = pd.concat([df_vae, new_row], ignore_index=True)

    # Write to CSV
    df_vae.to_csv(vae_stats_log, index=False)
    print(f"\nVAE statistics saved to: {vae_stats_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply VAE correction for masked basins')
    parser.add_argument('basins', nargs='+', help='Basin IDs to mask')
    parser.add_argument('--script_dir', type=str, required=True,
                        help='Directory where vae_statistics_log.csv will be saved')
    args = parser.parse_args()

    masked_basins = args.basins
    print(f"Received masked basins: {masked_basins}")

    # Run main function
    main(masked_basins, args.script_dir)

'''