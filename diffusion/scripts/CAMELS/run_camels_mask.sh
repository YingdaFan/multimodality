#!/bin/bash
# Generic CAMELS training and evaluation script (with basin mask support)
# Supports all diffusion models: NsDiff, TimeGrad, TimeDiff, D3VAE, DiffusionTS, etc.
#
# Usage:
#   bash run_camels_mask.sh NsDiff                           # NsDiff, without mask
#   bash run_camels_mask.sh NsDiff 01022500                  # NsDiff, mask a single basin
#   bash run_camels_mask.sh NsDiff 01022500 02069700         # NsDiff, mask multiple basins
#   bash run_camels_mask.sh TimeGrad 11141280 11143000       # TimeGrad, mask multiple basins

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash run_camels_mask.sh <MODEL_NAME> [basin_ids...]"
    echo ""
    echo "Supported models: NsDiff, TimeGrad, TimeDiff, D3VAE, DiffusionTS, CSDI, CSBI, SSSD"
    echo ""
    echo "Examples:"
    echo "  bash run_camels_mask.sh NsDiff"
    echo "  bash run_camels_mask.sh NsDiff 01022500 02069700"
    echo "  bash run_camels_mask.sh TimeGrad 11141280"
    exit 1
fi

# Get model name
MODEL_NAME=$1
shift

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
DATA_DIR="$(dirname $PROJECT_DIR)/data_processing"

cd "$PROJECT_DIR"
export PYTHONPATH=./

# Parse arguments - all remaining arguments are basin IDs (space-separated)
MASK_VALUES=("$@")
TARGET_BASIN="$@"

# Display configuration
echo "======================================"
echo "CAMELS Training: $MODEL_NAME"
if [ ${#MASK_VALUES[@]} -eq 0 ]; then
    echo "Mode: No Mask"
else
    echo "Mode: With Mask"
    echo "Mask basins: ${MASK_VALUES[*]}"
fi
echo "======================================"

# Step 1: Preprocess data (exclude masked basins from normalization calculation)
echo ""
echo "[Step 1/3] Preprocessing data..."
echo "======================================"

cd "$DATA_DIR"
if [ ${#MASK_VALUES[@]} -gt 0 ]; then
    echo "Excluding basins ${MASK_VALUES[*]} from Y normalization calculation..."
    python preprocess_allbasin_aligntime_camels.py "${MASK_VALUES[@]}"
else
    echo "Normal preprocessing (no basins excluded)..."
    python preprocess_allbasin_aligntime_camels.py
fi

cd "$PROJECT_DIR"
export PYTHONPATH=./

# Step 2: Train model
echo ""
echo "[Step 2/3] Training ${MODEL_NAME} model..."
echo "======================================"

# Build Python script path
PYTHON_SCRIPT="./src/experiments/${MODEL_NAME}_CAMELS.py"

# Check if the script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_CAMELS.py first."
    exit 1
fi

# Get batch size from prepped.npz (= number of basins)
NPZ_PATH="../data_processing/data/prepped.npz"
BATCH_SIZE=$(python3 -c "\
import numpy as np; \
print(int(np.load('$NPZ_PATH', allow_pickle=True)['n_segs']))")
echo "Batch size (n_segs): $BATCH_SIZE"

if [ ${#MASK_VALUES[@]} -eq 0 ]; then
    # Without mask
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 "$PYTHON_SCRIPT" \
        --dataset_type="CAMELS" \
        --npz_path="$NPZ_PATH" \
        --device="cuda" \
        --batch_size=$BATCH_SIZE \
        --horizon=1 \
        --pred_len=365 \
        --windows=365 \
        --load_pretrain=False \
        --epochs=200 \
        --patience=20 \
        --lr=0.001 \
        runs --seeds='[1]'
else
    # With mask: pass basin IDs (space-separated)
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 "$PYTHON_SCRIPT" \
        --dataset_type="CAMELS" \
        --npz_path="$NPZ_PATH" \
        --device="cuda" \
        --batch_size=$BATCH_SIZE \
        --horizon=1 \
        --pred_len=365 \
        --windows=365 \
        --load_pretrain=False \
        --epochs=200 \
        --patience=20 \
        --lr=0.001 \
        --masked_basin_ids ${MASK_VALUES[@]} \
        runs --seeds='[1]'
fi

# Step 3: Evaluate model
echo ""
echo "[Step 3/3] Evaluating predictions..."
echo "======================================"

cd "$DATA_DIR"

# Metrics log is stored in the project output directory
METRICS_LOG="$PROJECT_DIR/output/basin_metrics_log.csv"

if [ -z "$TARGET_BASIN" ]; then
    # No target_basin specified, evaluate all basins (without recording metrics log)
    python postprocess_perseg_aligntime.py \
        --pred_dir "$PROJECT_DIR/output/pred" \
        --partition trn
    # python postprocess_perseg_aligntime.py \
    #     --pred_dir "$PROJECT_DIR/output/pred" \
    #     --partition tst
else
    # Target_basin specified, evaluate and record metrics
    python postprocess_perseg_aligntime.py \
        --pred_dir "$PROJECT_DIR/output/pred" \
        --partition trn \
        --target_basin "$TARGET_BASIN" \
        --metrics_log "$METRICS_LOG"
    # python postprocess_perseg_aligntime.py \
    #     --pred_dir "$PROJECT_DIR/output/pred" \
    #     --partition tst \
    #     --target_basin "$TARGET_BASIN" \
    #     --metrics_log "$METRICS_LOG"
fi

echo ""
echo "======================================"
echo "Pipeline complete! (Model: $MODEL_NAME)"
if [ ${#MASK_VALUES[@]} -gt 0 ]; then
    echo "Masked basins: ${MASK_VALUES[*]}"
fi
echo "Predictions saved to: $PROJECT_DIR/output/pred/"
echo "Figures saved to: $PROJECT_DIR/output/figure/"
if [ -n "$TARGET_BASIN" ]; then
    echo "Metrics log: $METRICS_LOG"
fi
echo "======================================"
