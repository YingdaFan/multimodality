#!/bin/bash
set -e
# Stage 1 of LSTM+Diffusion RAW pipeline
#
# Data flow:
# 1. LSTM trains and outputs normalized predictions to .npy files (output/preds/)
# 2. Evaluate LSTM using postprocess_perseg_aligntime.py --use_vae_stats:
#    - Reads normalized predictions from preds/
#    - Denormalizes with y_mean_vae/y_std_vae
#    - Compares with y_raw_* (ground truth)
# 3. fill_prepped_npz_raw.py: y_obs_* ← denormalized values (for RAW Stage 2)
# 4. y_raw_* is NEVER modified (remains as ground truth)
#
# Key difference from run_camels_perstd_stage1.sh:
#   Uses --use_vae_stats for denormalization, simulating real application
#   where true y_mean/y_std are unknown and must be estimated by VAE.
#
# Usage: bash run_camels_perstd_stage1_raw.sh <target_basins...>

if [ $# -eq 0 ]; then
    echo "Usage: bash run_camels_perstd_stage1_raw.sh <target_basins...>"
    echo "Example: bash run_camels_perstd_stage1_raw.sh 01022500 01031500"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMPUTATION_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$IMPUTATION_DIR/data_processing"

TARGET_TEST_BASINS="$@"

echo "========================================"
echo "Stage 1: LSTM Training (RAW Pipeline)"
echo "========================================"
echo "Target Basin(s): $TARGET_TEST_BASINS"
echo "========================================"

# Step 1: Preprocessing (per-basin standardization)
echo ""
echo "[Step 1/6] Preprocessing data..."
cd "$DATA_DIR"
python ${PREPROCESS_SCRIPT:-preprocess_camelsh_multimodality.py}

# Step 2: Mask target basins
echo ""
echo "[Step 2/6] Masking basins in training data..."
python modify_basin_to_nan_allmask.py $TARGET_TEST_BASINS

# Step 3: Apply VAE
echo ""
echo "[Step 3/6] Applying VAE correction for masked basins..."
python apply_vae.py $TARGET_TEST_BASINS --script_dir "$SCRIPT_DIR"

# Step 4: Train LSTM
echo ""
echo "[Step 4/6] Training LSTM model..."
cd "$SCRIPT_DIR"
python base.py

# Step 5: Evaluate LSTM results
# Uses --use_vae_stats to denormalize with y_mean_vae/y_std_vae
echo ""
echo "[Step 5/6] Evaluating LSTM predictions (using VAE stats)..."
METRICS_LOG="$SCRIPT_DIR/output/basin_metrics_log.csv"

cd "$DATA_DIR"

# Evaluate training set
python postprocess_perseg_aligntime.py \
    --pred_dir "$SCRIPT_DIR/output/preds" \
    --model_name "LSTM" \
    --partition trn \
    --target_basin "$TARGET_TEST_BASINS" \
    --metrics_log "$METRICS_LOG" \
    --use_vae_stats

# Evaluate test set
python postprocess_perseg_aligntime.py \
    --pred_dir "$SCRIPT_DIR/output/preds" \
    --model_name "LSTM" \
    --partition tst \
    --target_basin "$TARGET_TEST_BASINS" \
    --metrics_log "$METRICS_LOG" \
    --use_vae_stats

# Merge VAE statistics
echo ""
echo "Merging basin metrics with VAE statistics..."
python merge_basin_metrics.py --script_dir "$SCRIPT_DIR"

# Step 6: Fill y_obs_* with denormalized values for RAW Stage 2
echo ""
echo "[Step 6/6] Preparing y_obs_* for RAW Stage 2..."
cd "$SCRIPT_DIR"
python fill_prepped_npz_raw.py $TARGET_TEST_BASINS

echo ""
echo "========================================"
echo "Stage 1 (RAW) Complete!"
echo "========================================"
echo "  - LSTM trained and evaluated"
echo "  - Predictions denormalized with y_mean_vae/y_std_vae"
echo "  - y_obs_* filled for Stage 2 input"
echo "  - y_raw_* unchanged (ground truth)"
echo "========================================"
