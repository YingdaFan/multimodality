#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():

    print("VAE Hyperparameter Search Results Analysis - Balanced Version")

    print()

    # 1. Load data
    csv_file = 'hyperparam_search.csv'

    if not Path(csv_file).exists():
        print(f"Results file not found: {csv_file}")
        print("   Please run hyperparameter search first: bash run_hyper_balanced.sh")
        return

    df = pd.read_csv(csv_file)
    print(f"Loaded data: {csv_file}")
    print(f"   Total {len(df)} records, {df['hyperparam_id'].nunique()} hyperparameter combinations\n")

    # 2. Aggregate results
    print("Aggregating hyperparameter results...")

    agg_cols = {
        'mean_error_pct': 'mean',
        'std_error_pct': 'mean',
    }

    grouped = df.groupby('hyperparam_id').agg(agg_cols).reset_index()

    # Add hyperparameter columns
    hyperparam_cols = ['latent_dim', 'hidden_dim', 'lr', 'dropout', 'epochs',
                      'beta_schedule', 'beta_value', 'batch_size']

    for col in hyperparam_cols:
        if col in df.columns:
            first_records = df.drop_duplicates('hyperparam_id')[['hyperparam_id', col]]
            grouped = grouped.merge(first_records, on='hyperparam_id', how='left')

    print(f"   Total {len(grouped)} hyperparameter combinations (expected 256)\n")

    # 3. Find best hyperparameters

    print("Best hyperparameter combinations")

    print()

    df_sorted = grouped.sort_values('mean_error_pct')

    # Top 10
    print("Top 10 best hyperparameter combinations:")


    display_cols = ['hyperparam_id', 'mean_error_pct', 'std_error_pct']
    for col in hyperparam_cols:
        if col in df_sorted.columns:
            display_cols.append(col)

    print(df_sorted[display_cols].head(10).to_string(index=False))
    print()

    # Best combination
    best = df_sorted.iloc[0]
    print("Best hyperparameter combination (ID={}):".format(int(best['hyperparam_id'])))


    for col in hyperparam_cols:
        if col in best.index:
            print(f"  {col:20s}: {best[col]}")

    print()
    print(f"  Average mean error: {best['mean_error_pct']:.2f}%")
    print(f"  Average std error: {best['std_error_pct']:.2f}%")
    print()

    # 4. Visualization
    print("Generating visualization plots...")

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('VAE Hyperparameter Search Results (Balanced)', fontsize=16, y=0.995)

    # 1. Learning Rate
    ax = axes[0, 0]
    if 'lr' in grouped.columns:
        lr_grouped = grouped.groupby('lr')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        ax.errorbar(lr_grouped['lr'], lr_grouped['mean'], yerr=lr_grouped['std'],
                   marker='o', markersize=10, capsize=8, capthick=2, linewidth=2.5)
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Learning Rate vs Error', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    # 2. Dropout
    ax = axes[0, 1]
    if 'dropout' in grouped.columns:
        dropout_grouped = grouped.groupby('dropout')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        ax.errorbar(dropout_grouped['dropout'], dropout_grouped['mean'], yerr=dropout_grouped['std'],
                   marker='s', markersize=10, capsize=8, capthick=2, linewidth=2.5, color='orange')
        ax.set_xlabel('Dropout Rate', fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Dropout Rate vs Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 3. Latent Dimension
    ax = axes[0, 2]
    if 'latent_dim' in grouped.columns:
        latent_grouped = grouped.groupby('latent_dim')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        ax.errorbar(latent_grouped['latent_dim'], latent_grouped['mean'], yerr=latent_grouped['std'],
                   marker='D', markersize=10, capsize=8, capthick=2, linewidth=2.5, color='green')
        ax.set_xlabel('Latent Dimension', fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Latent Dimension vs Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 4. Hidden Dimension
    ax = axes[1, 0]
    if 'hidden_dim' in grouped.columns:
        hidden_grouped = grouped.groupby('hidden_dim')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        ax.errorbar(hidden_grouped['hidden_dim'], hidden_grouped['mean'], yerr=hidden_grouped['std'],
                   marker='^', markersize=10, capsize=8, capthick=2, linewidth=2.5, color='purple')
        ax.set_xlabel('Hidden Dimension', fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Hidden Dimension vs Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 5. Epochs
    ax = axes[1, 1]
    if 'epochs' in grouped.columns:
        epochs_grouped = grouped.groupby('epochs')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        ax.errorbar(epochs_grouped['epochs'], epochs_grouped['mean'], yerr=epochs_grouped['std'],
                   marker='v', markersize=10, capsize=8, capthick=2, linewidth=2.5, color='red')
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Epochs vs Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 6. Batch Size
    ax = axes[1, 2]
    if 'batch_size' in grouped.columns:
        batch_grouped = grouped.groupby('batch_size')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        ax.errorbar(batch_grouped['batch_size'], batch_grouped['mean'], yerr=batch_grouped['std'],
                   marker='p', markersize=10, capsize=8, capthick=2, linewidth=2.5, color='brown')
        ax.set_xlabel('Batch Size', fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Batch Size vs Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 7. Beta Schedule
    ax = axes[2, 0]
    if 'beta_schedule' in grouped.columns:
        schedule_grouped = grouped.groupby('beta_schedule')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        x_pos = np.arange(len(schedule_grouped))
        ax.bar(x_pos, schedule_grouped['mean'], yerr=schedule_grouped['std'],
              capsize=8, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(schedule_grouped['beta_schedule'])
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Beta Schedule vs Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    # 8. Beta Value
    ax = axes[2, 1]
    if 'beta_value' in grouped.columns:
        beta_grouped = grouped.groupby('beta_value')['mean_error_pct'].agg(['mean', 'std']).reset_index()
        ax.errorbar(beta_grouped['beta_value'], beta_grouped['mean'], yerr=beta_grouped['std'],
                   marker='h', markersize=10, capsize=8, capthick=2, linewidth=2.5, color='navy')
        ax.set_xlabel('Beta Value', fontsize=12)
        ax.set_ylabel('Mean Error (%)', fontsize=12)
        ax.set_title('Beta Value vs Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 9. Top 10 combination comparison
    ax = axes[2, 2]
    top10 = df_sorted.head(10)
    x_pos = np.arange(len(top10))
    ax.bar(x_pos, top10['mean_error_pct'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"ID{int(i)}" for i in top10['hyperparam_id']], rotation=45)
    ax.set_ylabel('Mean Error (%)', fontsize=12)
    ax.set_title('Top 10 Combinations', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = 'hyperparam_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.show()

    # 5. Save top results
    output_file = 'top_hyperparams.csv'
    df_sorted.head(20).to_csv(output_file, index=False)
    print(f"\nSaved top 20 results: {output_file}")


    print("Analysis complete!")



if __name__ == "__main__":
    main()
