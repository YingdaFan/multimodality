#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

# File paths
script_dir = Path(__file__).parent
input_csv = script_dir / "hyperparam_search.csv"
output_csv = script_dir / "hyperparam_search_final.csv"

# Load raw results
print(f"Loading data: {input_csv}")
df = pd.read_csv(input_csv)



# Group by hyperparameter combination and aggregate
print("Aggregating data...")
agg_funcs = {
    # Preserve hyperparameter values (take first since same combination has same values)
    'latent_dim': 'first',
    'hidden_dim': 'first',
    'lr': 'first',
    'dropout': 'first',
    'epochs': 'first',
    'beta_schedule': 'first',
    'beta_value': 'first',
    'batch_size': 'first',

    # Statistical metrics
    'mean_error_pct': ['mean', 'std', 'min', 'max'],
    'std_error_pct': ['mean', 'std', 'min', 'max'],

    # Record number of tested basins
    'basin': 'count'
}

df_agg = df.groupby('hyperparam_id').agg(agg_funcs).reset_index()

# Flatten column names
df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                  for col in df_agg.columns.values]

# Rename columns for clarity
df_agg.rename(columns={
    'latent_dim_first': 'latent_dim',
    'hidden_dim_first': 'hidden_dim',
    'lr_first': 'lr',
    'dropout_first': 'dropout',
    'epochs_first': 'epochs',
    'beta_schedule_first': 'beta_schedule',
    'beta_value_first': 'beta_value',
    'batch_size_first': 'batch_size',
    'mean_error_pct_mean': 'avg_mean_error_pct',
    'mean_error_pct_std': 'std_mean_error_pct',
    'mean_error_pct_min': 'min_mean_error_pct',
    'mean_error_pct_max': 'max_mean_error_pct',
    'std_error_pct_mean': 'avg_std_error_pct',
    'std_error_pct_std': 'std_std_error_pct',
    'std_error_pct_min': 'min_std_error_pct',
    'std_error_pct_max': 'max_std_error_pct',
    'basin_count': 'num_basins_tested'
}, inplace=True)

# Sort by average error
df_agg = df_agg.sort_values('avg_mean_error_pct')

# Save results
print(f"Saving aggregated results: {output_csv}")
df_agg.to_csv(output_csv, index=False, float_format='%.4f')

print(f"\nAggregation complete!")
print(f"Output file: {output_csv}")
print(f"Total {len(df_agg)} hyperparameter combinations")
print()



display_cols = [
    'hyperparam_id',
    'latent_dim', 'hidden_dim', 'lr', 'dropout',
    'batch_size', 'beta_schedule', 'beta_value',
    'avg_mean_error_pct', 'avg_std_error_pct',
    'num_basins_tested'
]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
pd.set_option('display.float_format', '{:.2f}'.format)

print(df_agg[display_cols].head(10).to_string(index=False))
print()

