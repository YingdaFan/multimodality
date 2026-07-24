#!/bin/bash
set -e
# Single-fold MIDM baseline: train + evaluate
#
# Usage: bash run_midm_stage.sh <target_basins...>
#
# Example:
#   bash run_midm_stage.sh 01022500 02069700

if [ $# -lt 1 ]; then
    echo "Usage: bash run_midm_stage.sh <target_basins...>"
    exit 1
fi

TARGET_BASINS=("$@")
TARGET_BASIN="$@"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MIDM_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$(dirname $MIDM_DIR)/data_processing"
NPZ_PATH="$DATA_DIR/data/prepped.npz"

echo "======================================"
echo "MIDM Spatial Extrapolation Baseline"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo ""

# --------------------------------------------------
# Step 1: Train MIDM
# --------------------------------------------------
echo "[Step 1/2] Training MIDM ..."
cd "$MIDM_DIR"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 train.py \
    --npz_path="$NPZ_PATH" \
    --device="cuda" \
    --hidden_dim=128 \
    --n_heads=4 \
    --n_layers=4 \
    --n_diffusion_steps=50 \
    --max_locations=64 \
    --max_locations_infer=128 \
    --n_repeats=8 \
    --epochs=200 \
    --patience=20 \
    --lr=0.001 \
    --n_pred_samples=10 \
    --n_pred_steps=50 \
    --masked_basins ${TARGET_BASINS[@]}

# --------------------------------------------------
# Step 2: Evaluate
# --------------------------------------------------
echo ""
echo "[Step 2/2] Evaluating predictions ..."
cd "$DATA_DIR"

METRICS_LOG="$MIDM_DIR/output/basin_metrics_log.csv"

python3 postprocess_perseg_aligntime_raw.py \
    --pred_dir "$MIDM_DIR/output/pred" \
    --model_name "midm" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "MIDM Baseline Complete!"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions: $MIDM_DIR/output/pred/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
