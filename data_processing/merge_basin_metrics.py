#!/usr/bin/env python3
"""
Merge basin_metrics_log_trn.csv / basin_metrics_log_tst.csv with vae_statistics_log.csv
to generate basin_metrics_combined_trn.csv / basin_metrics_combined_tst.csv with complete info
"""

import os
import pandas as pd
import sys
import argparse


def merge_metrics_for_partition(script_dir, partition):
    """
    Merge model evaluation metrics with VAE statistics (for specified partition)

    Args:
        script_dir: Directory where the script resides (e.g., lstm/)
        partition: 'trn' or 'tst'
    """
    # Use the provided script_dir (i.e., the directory of the calling shell script, e.g., lstm/)
    # Define file paths (all under the output/ directory)
    output_dir = os.path.join(script_dir, 'output')
    metrics_log = os.path.join(output_dir, f'basin_metrics_log_{partition}.csv')
    vae_stats_log = os.path.join(output_dir, 'vae_statistics_log.csv')
    combined_log = os.path.join(output_dir, f'basin_metrics_combined_{partition}.csv')

    # Check if files exist
    if not os.path.exists(metrics_log):
        print(f"Warning: {metrics_log} not found, skipping merge.")
        return

    if not os.path.exists(vae_stats_log):
        print(f"Warning: {vae_stats_log} not found, skipping merge.")
        print("  (This is expected if no VAE correction was applied)")
        return

    # Read both CSV files
    print(f"Reading {metrics_log}...")
    df_metrics = pd.read_csv(metrics_log)

    print(f"Reading {vae_stats_log}...")
    df_vae = pd.read_csv(vae_stats_log)

    # Merge by basin (using the latest records)
    # Since both files may have multiple records for the same basin, we need to deduplicate and keep the latest
    print("\nMerging dataframes by basin...")

    # For metrics, keep the latest record for each basin
    df_metrics_sorted = df_metrics.sort_values('timestamp', ascending=False)
    df_metrics_latest = df_metrics_sorted.drop_duplicates(subset=['basin'], keep='first')

    # For VAE stats, keep the latest record for each basin
    df_vae_sorted = df_vae.sort_values('timestamp', ascending=False)
    df_vae_latest = df_vae_sorted.drop_duplicates(subset=['basin'], keep='first')

    # Perform left join: metrics as primary, adding VAE statistics
    df_combined = pd.merge(
        df_metrics_latest,
        df_vae_latest,
        on='basin',
        how='left',
        suffixes=('_metrics', '_vae')
    )

    # Reorder columns for better readability
    column_order = [
        'timestamp_metrics',  # Keep metrics timestamp as the primary timestamp
        'basin',
        'nse',
        'kge',
        'rmse',
        'mae',
        'r2',
        'pbias',
        'y_mean_true',
        'y_mean_vae',
        'y_std_true',
        'y_std_vae',
        'mean_error_pct',
        'std_error_pct'
    ]

    # Only select columns that exist
    available_columns = [col for col in column_order if col in df_combined.columns]
    df_combined = df_combined[available_columns]

    # Rename timestamp column
    if 'timestamp_metrics' in df_combined.columns:
        df_combined = df_combined.rename(columns={'timestamp_metrics': 'timestamp'})

    # Save the merged CSV
    df_combined.to_csv(combined_log, index=False)


    print(f"   Combined file saved to: {combined_log}")

    # Display sample rows
    if len(df_combined) > 0:
        print("\nSample combined data:")
        print(df_combined.head(3).to_string(index=False))


def merge_metrics(script_dir):
    """
    Merge metrics for all partitions (trn and tst)
    """


    merge_metrics_for_partition(script_dir, 'trn')

    merge_metrics_for_partition(script_dir, 'tst')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge basin metrics and VAE statistics')
    parser.add_argument('--script_dir', type=str, required=True,
                        help='Directory where basin_metrics_log_trn.csv, basin_metrics_log_tst.csv and vae_statistics_log.csv are located')
    args = parser.parse_args()

    try:
        merge_metrics(args.script_dir)
    except Exception as e:
        print(f"Error during merge: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
