#!/usr/bin/env python


import os
import sys
import yaml
import subprocess
import pandas as pd
import numpy as np
import itertools
import argparse
from pathlib import Path
from datetime import datetime
import json

# Get script directory
SCRIPT_DIR = Path(__file__).parent
IMPUTATION_DIR = SCRIPT_DIR.parent
TEMPORAL_DIR = IMPUTATION_DIR.parent

# CSV file path (consistent with run_camels_kfold.sh)
CSV_FILE = TEMPORAL_DIR / 'denormalized_camels_data_time.parquet'
NUM_FOLDS = 106
NUM_HYPERPARAM_FOLDS = 2  # First 2 folds for hyperparameter tuning


def extract_validation_basins(csv_file=CSV_FILE, num_folds=NUM_FOLDS, num_hyperparam_folds=NUM_HYPERPARAM_FOLDS):
    """
    Dynamically extract basins from the first N folds for hyperparameter validation.
    Reuses the fold logic from run_camels_kfold.sh for consistency.

    Args:
        csv_file: Path to the CSV data file
        num_folds: Total number of folds (default 106)
        num_hyperparam_folds: Number of folds for hyperparameter tuning (default 2)

    Returns:
        list of basin IDs for hyperparameter validation
    """
    print(f"Extracting validation basins from {csv_file}...")

    # Read CSV and extract unique basin IDs (column: basin_id)
    # Use dtype to preserve leading zeros
    df = pd.read_parquet(csv_file)
    all_basins = sorted(df['basin_id'].unique())
    total_basins = len(all_basins)

    print(f"Total basins found: {total_basins}")

    # Validate fold count
    if num_folds > total_basins:
        raise ValueError(f"NUM_FOLDS ({num_folds}) cannot be greater than TOTAL_BASINS ({total_basins})")

    # Compute fold size (consistent with run_camels_kfold.sh logic)
    basins_per_fold = total_basins // num_folds
    remainder = total_basins % num_folds

    print(f"Basins per fold: ~{basins_per_fold}")
    if remainder > 0:
        print(f"Note: First {remainder} folds will have {basins_per_fold + 1} basins")

    # Compute how many basins the first N folds contain
    hyperparam_basin_count = 0
    for fold in range(1, num_hyperparam_folds + 1):
        if fold <= remainder:
            hyperparam_basin_count += basins_per_fold + 1
        else:
            hyperparam_basin_count += basins_per_fold

    # Extract the first hyperparam_basin_count basins
    validation_basins = all_basins[:hyperparam_basin_count]

    print(f"Extracted {len(validation_basins)} validation basins (folds 1-{num_hyperparam_folds})")
    print(f"Validation basins: {validation_basins[:5]}... (showing first 5)")

    return validation_basins


PARAM_GRID = {
    'ft_epochs': [50, 100, 150],                          # 3 values, few to many
    'finetune_learning_rate': [0.001, 0.005, 0.01],       # 3 values, spread apart
    'weight_decay': [0.0, 1e-5, 1e-4, 1e-3],              # 4 values, L2 regularization strength
}




def load_config():
    """Load current configuration."""
    config_path = SCRIPT_DIR / 'config.yml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def update_config(params):
    """Update config.yml with new hyperparameters."""
    config_path = SCRIPT_DIR / 'config.yml'

    # Read current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update hyperparameters
    for key, value in params.items():
        config[key] = value

    # Write back
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Updated config.yml with: {params}")


def grid_search(param_grid):
    """Generate all possible hyperparameter combinations (grid search)."""
    keys = param_grid.keys()
    values = param_grid.values()
    for v in itertools.product(*values):
        yield dict(zip(keys, v))


def run_single_experiment(params, basin_ids, experiment_id):
    """Run a single experiment (train on all basins except validation basins, test on validation basins).

    Args:
        params: dict of hyperparameters
        basin_ids: list of basin IDs to use as validation/test basins (will be masked)
        experiment_id: unique identifier for this experiment

    Returns:
        dict with experiment results including metrics
    """

    # Update config file
    update_config(params)

    # Call run_camels_perstd.sh with validation basins
    cmd = ['bash', str(SCRIPT_DIR / 'run_camels_perstd.sh')] + basin_ids

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return None

        print("Training completed successfully!")

    except subprocess.TimeoutExpired:
        print("ERROR: Training timed out after 1 hour")
        return None
    except Exception as e:
        print(f"ERROR: Exception during training: {e}")
        return None

    # Read evaluation metrics
    metrics_file = SCRIPT_DIR / 'basin_metrics_log.csv'

    if not metrics_file.exists():
        print(f"ERROR: Metrics file not found: {metrics_file}")
        return None

    try:
        metrics_df = pd.read_csv(metrics_file)

        # Only take this experiment's basin results (last N rows)
        if len(metrics_df) >= len(basin_ids):
            recent_metrics = metrics_df.tail(len(basin_ids))
        else:
            recent_metrics = metrics_df

        # Compute average metrics
        result = {
            **params,
            'experiment_id': experiment_id,
            'num_basins': len(basin_ids),
            'avg_rmse': recent_metrics['rmse'].mean(),
            'std_rmse': recent_metrics['rmse'].std(),
            'avg_nse': recent_metrics['nse'].mean(),
            'std_nse': recent_metrics['nse'].std(),
            'avg_pbias': recent_metrics['pbias'].mean(),
            'std_pbias': recent_metrics['pbias'].std(),
            'min_rmse': recent_metrics['rmse'].min(),
            'max_rmse': recent_metrics['rmse'].max(),
            'min_nse': recent_metrics['nse'].min(),
            'max_nse': recent_metrics['nse'].max(),
        }

        print(f"\nResults:")
        print(f"  Avg RMSE: {result['avg_rmse']:.4f} +/- {result['std_rmse']:.4f}")
        print(f"  Avg NSE:  {result['avg_nse']:.4f} +/- {result['std_nse']:.4f}")
        print(f"  Avg PBIAS: {result['avg_pbias']:.4f} +/- {result['std_pbias']:.4f}")

        return result

    except Exception as e:
        print(f"ERROR: Failed to read metrics: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter grid search for LSTM imputation')
    parser.add_argument('--output_dir', type=str, default='hyperparam_results',
                        help='Directory to save results')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous results file')

    args = parser.parse_args()

    # Dynamically extract basins from first 2 folds for validation (consistent with run_camels_kfold.sh)
    print("\nSTEP 1: Extracting validation basins")
    validation_basins = extract_validation_basins()
    print()

    # Use full parameter space
    param_grid = PARAM_GRID

    # Create output directory
    output_dir = SCRIPT_DIR / args.output_dir
    output_dir.mkdir(exist_ok=True)

    # Results file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'hyperparameter_results_{timestamp}.csv'

    # If resuming, load previous results
    existing_results = []
    if args.resume:
        existing_files = sorted(output_dir.glob('hyperparameter_results_*.csv'))
        if existing_files:
            latest_file = existing_files[-1]
            print(f"Resuming from: {latest_file}")
            existing_results = pd.read_csv(latest_file).to_dict('records')
            results_file = latest_file  # Continue writing to the same file

    # Generate all hyperparameter combinations (grid search)
    param_combinations = list(grid_search(param_grid))
    print(f"\nGrid Search: {len(param_combinations)} combinations")
    print(f"\nSearch space:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    print()

    # Run experiments
    results = existing_results.copy()

    for i, params in enumerate(param_combinations, start=len(existing_results)+1):
        experiment_id = f"exp_{timestamp}_{i:03d}"
        print(f"\nExperiment {i}/{len(param_combinations)}: {experiment_id}")

        result = run_single_experiment(
            params=params,
            basin_ids=validation_basins,
            experiment_id=experiment_id,
        )

        if result is not None:
            results.append(result)

            # Save intermediate results
            pd.DataFrame(results).to_csv(results_file, index=False)
            print(f"\nSaved results to: {results_file}")
        else:
            print(f"\nSkipping experiment {i} due to errors")

    # Final analysis
    results_df = pd.DataFrame(results)

    # Find best parameters (by avg_rmse)
    best_idx = results_df['avg_rmse'].idxmin()
    best_result = results_df.loc[best_idx]

    print("\nBEST PARAMETERS (lowest avg RMSE):")
    for key in ['ft_epochs', 'finetune_learning_rate', 'weight_decay']:
        if key in best_result:
            print(f"  {key:25s}: {best_result[key]}")
    print(f"\n  avg_rmse: {best_result['avg_rmse']:.4f} +/- {best_result['std_rmse']:.4f}")
    print(f"  avg_nse:  {best_result['avg_nse']:.4f} +/- {best_result['std_nse']:.4f}")
    print(f"  avg_pbias: {best_result['avg_pbias']:.4f} +/- {best_result['std_pbias']:.4f}")

    # Find best parameters (by avg_nse)
    best_nse_idx = results_df['avg_nse'].idxmax()
    best_nse_result = results_df.loc[best_nse_idx]

    print("\nBEST PARAMETERS (highest avg NSE):")
    for key in ['ft_epochs', 'finetune_learning_rate', 'weight_decay']:
        if key in best_nse_result:
            print(f"  {key:25s}: {best_nse_result[key]}")
    print(f"\n  avg_nse:  {best_nse_result['avg_nse']:.4f} +/- {best_nse_result['std_nse']:.4f}")
    print(f"  avg_rmse: {best_nse_result['avg_rmse']:.4f} +/- {best_nse_result['std_rmse']:.4f}")
    print(f"  avg_pbias: {best_nse_result['avg_pbias']:.4f} +/- {best_nse_result['std_pbias']:.4f}")

    # Save best config to file
    best_config_file = output_dir / f'best_config_{timestamp}.yml'
    best_config = {k: best_result[k] for k in ['ft_epochs', 'finetune_learning_rate', 'weight_decay']
                   if k in best_result}

    with open(best_config_file, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)

    print(f"\n\nBest configuration saved to: {best_config_file}")
    print(f"All results saved to: {results_file}")

    print("\nTOP 5 CONFIGURATIONS (by NSE):\n")
    top5 = results_df.nlargest(5, 'avg_nse')
    print(top5[['ft_epochs', 'finetune_learning_rate', 'weight_decay',
                'avg_nse', 'avg_rmse']].to_string(index=False))

    print(f"\n\nTo use the best hyperparameters, run:")
    print(f"  cp {best_config_file} config.yml")
    print(f"  bash run_camels_kfold.sh 106")


if __name__ == '__main__':
    main()
