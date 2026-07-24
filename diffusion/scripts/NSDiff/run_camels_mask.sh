#!/bin/bash
# NsDiff CAMELS training and evaluation script (with basin mask support)
#
# Usage:
#   bash run_camels_mask.sh                          # without mask
#   bash run_camels_mask.sh 01022500                 # mask a single basin
#   bash run_camels_mask.sh 01022500 02069700        # mask multiple basins (space-separated)

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
DATA_DIR="$(dirname $PROJECT_DIR)/data_processing"

cd "$PROJECT_DIR"
export PYTHONPATH=./

# Parse arguments - all arguments are basin IDs (space-separated)
MASK_VALUES=("$@")
TARGET_BASIN="$@"

# Display configuration
echo "======================================"
if [ ${#MASK_VALUES[@]} -eq 0 ]; then
    echo "NsDiff CAMELS Training (No Mask)"
else
    echo "NsDiff CAMELS Training (With Mask)"
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
echo "[Step 2/3] Training NsDiff model..."
echo "======================================"

if [ ${#MASK_VALUES[@]} -eq 0 ]; then
    # Without mask
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 ./src/experiments/NsDiff_CAMELS.py \
        --dataset_type="CAMELS" \
        --npz_path="../data_processing/data/prepped.npz" \
        --device="cuda" \
        --batch_size=32 \
        --horizon=1 \
        --pred_len=365 \
        --windows=365 \
        --load_pretrain=False \
        --epochs=300 \
        --patience=10 \
        --lr=0.001 \
        runs --seeds='[1]'
else
    # With mask: pass basin IDs (space-separated)
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 ./src/experiments/NsDiff_CAMELS.py \
        --dataset_type="CAMELS" \
        --npz_path="../data_processing/data/prepped.npz" \
        --device="cuda" \
        --batch_size=32 \
        --horizon=1 \
        --pred_len=365 \
        --windows=365 \
        --load_pretrain=False \
        --epochs=300 \
        --patience=10 \
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
    python postprocess_perseg_aligntime.py \
        --pred_dir "$PROJECT_DIR/output/pred" \
        --partition tst
else
    # Target_basin specified, evaluate and record metrics
    python postprocess_perseg_aligntime.py \
        --pred_dir "$PROJECT_DIR/output/pred" \
        --partition trn \
        --target_basin "$TARGET_BASIN" \
        --metrics_log "$METRICS_LOG"
    python postprocess_perseg_aligntime.py \
        --pred_dir "$PROJECT_DIR/output/pred" \
        --partition tst \
        --target_basin "$TARGET_BASIN" \
        --metrics_log "$METRICS_LOG"
fi

echo ""
echo "======================================"
echo "Pipeline complete!"
if [ ${#MASK_VALUES[@]} -gt 0 ]; then
    echo "Masked basins: ${MASK_VALUES[*]}"
fi
echo "Predictions saved to: $PROJECT_DIR/output/pred/"
echo "Figures saved to: $PROJECT_DIR/output/figure/"
if [ -n "$TARGET_BASIN" ]; then
    echo "Metrics log: $METRICS_LOG"
fi
echo "======================================"
