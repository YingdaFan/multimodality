#!/bin/bash
set -e
# Single-fold MIDM Calibration: LSTM (Stage 1) + MIDM SDEdit (Stage 2)
#
# Stage 1: Reuses run_camels_perstd_stage1_raw.sh (LSTM + VAE + fill NPZ)
# Stage 2: MIDM calibration with SDEdit sampling
#
# Usage: bash run_midm_cal_stage.sh <target_basins...>

if [ $# -lt 1 ]; then
    echo "Usage: bash run_midm_cal_stage.sh <target_basins...>"
    exit 1
fi

TARGET_BASINS=("$@")
TARGET_BASIN="$@"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MIDM_DIR="$(dirname $SCRIPT_DIR)"
IMPUTATION_DIR="$(dirname $MIDM_DIR)"
LSTM_DIR="$IMPUTATION_DIR/lstm"
DATA_DIR="$IMPUTATION_DIR/data_processing"
NPZ_PATH="$DATA_DIR/data/prepped.npz"

echo "======================================"
echo "MIDM Calibration (LSTM + SDEdit)"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo ""

# --------------------------------------------------
# Stage 1: LSTM (preprocess → mask → VAE → LSTM → fill NPZ)
# --------------------------------------------------
echo "------------------------------------------"
echo "Stage 1: LSTM Training & RAW NPZ Fill"
echo "------------------------------------------"
bash "$LSTM_DIR/run_camels_perstd_stage1_raw.sh" ${TARGET_BASINS[@]}

# --------------------------------------------------
# Stage 2: MIDM Calibration (SDEdit)
# --------------------------------------------------
echo ""
echo "------------------------------------------"
echo "Stage 2: MIDM Calibration (SDEdit)"
echo "------------------------------------------"
cd "$MIDM_DIR"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 train_cal.py \
    --npz_path="$NPZ_PATH" \
    --device="cuda" \
    --d_model=128 \
    --n_heads=4 \
    --n_layers=3 \
    --n_diffusion_steps=50 \
    --cov_rank=8 \
    --n_repeats=8 \
    --epochs=200 \
    --patience=30 \
    --lr=0.001 \
    --n_pred_samples=10 \
    --t_start_frac=0.33 \
    --dropout=0.1 \
    --masked_basins ${TARGET_BASINS[@]}

# --------------------------------------------------
# Stage 3: Evaluate
# --------------------------------------------------
echo ""
echo "[Evaluating predictions ...]"
cd "$DATA_DIR"

METRICS_LOG="$MIDM_DIR/output/basin_metrics_log.csv"

python3 postprocess_perseg_aligntime_raw.py \
    --pred_dir "$MIDM_DIR/output/pred" \
    --model_name "midm_cal" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "MIDM Calibration Complete!"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions: $MIDM_DIR/output/pred/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
