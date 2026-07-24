#!/bin/bash
# Stage 2 of LSTM+Diffusion pipeline
# Uses prepped.npz filled by Stage 1 (LSTM predictions for masked basins)
#
# Key differences from run_camels_mask.sh:
#   1. Skips preprocessing (uses filled prepped.npz from Stage 1)
#   2. Does NOT use --masked_basin_ids (treats LSTM predictions as real data)
#   3. Still evaluates with --target_basin to get metrics for original masked basins
#
# Usage: bash run_camels_stage2.sh <MODEL_NAME> <target_basins...>
#
# Example:
#   bash run_camels_stage2.sh NsDiff 01022500 02069700

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash run_camels_stage2.sh <MODEL_NAME> <target_basins...>"
    echo ""
    echo "Supported models: NsDiff, TimeGrad, TimeDiff, D3VAE, DiffusionTS, CSDI, CSBI, SSSD"
    echo ""
    echo "This script is Stage 2 of the LSTM+Diffusion pipeline."
    echo "It assumes prepped.npz has been filled with LSTM predictions by Stage 1."
    echo ""
    echo "Example:"
    echo "  bash run_camels_stage2.sh NsDiff 01022500 02069700"
    exit 1
fi

# Get model name
MODEL_NAME=$1
shift

# Get target basins (for evaluation only, NOT for masking)
TARGET_BASINS=("$@")
TARGET_BASIN="$@"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
DATA_DIR="$(dirname $PROJECT_DIR)/data_processing"

cd "$PROJECT_DIR"
export PYTHONPATH=./

# Display configuration
echo "======================================"
echo "Stage 2: Diffusion Refinement"
echo "======================================"
echo "Model: $MODEL_NAME"
echo "Target basins (for evaluation): ${TARGET_BASINS[*]}"
echo ""
echo "NOTE: Using prepped.npz filled by Stage 1 (LSTM)"
echo "      Diffusion will treat LSTM predictions as real data"
echo "======================================"

# Skip Step 1: Preprocessing
# We use the prepped.npz that was filled by Stage 1 (LSTM predictions)
echo ""
echo "[Step 1/2] Skipping preprocessing (using Stage 1 filled prepped.npz)"
echo "======================================"

# Step 2: Train diffusion model
# IMPORTANT: Do NOT use --masked_basin_ids
# This allows the model to use LSTM predictions as input (autoregressive)
echo ""
echo "[Step 2/2] Training ${MODEL_NAME} model..."
echo "======================================"

# Build Python script path
PYTHON_SCRIPT="./src/experiments/${MODEL_NAME}_CAMELS.py"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_CAMELS.py first."
    exit 1
fi

# Train WITHOUT --masked_basin_ids
# Get batch size from prepped.npz (= number of basins)
NPZ_PATH="../data_processing/data/prepped.npz"
BATCH_SIZE=$(python3 -c "\
import numpy as np; \
print(int(np.load('$NPZ_PATH', allow_pickle=True)['n_segs']))")
echo "Batch size (n_segs): $BATCH_SIZE"

# Auto-detect seq_len from prepped.npz (uses y_obs_trn -- ~1MB, much smaller
# than x_trn). Env var still wins if set; falls back to 365 if detection fails.
DETECTED_SEQ_LEN=$(python3 -c "\
import numpy as np; \
print(int(np.load('$NPZ_PATH', allow_pickle=True)['y_obs_trn'].shape[1]))" 2>/dev/null)
PRED_LEN=${DIFFUSION_PRED_LEN:-${DETECTED_SEQ_LEN:-365}}
WINDOWS=${DIFFUSION_WINDOWS:-${DETECTED_SEQ_LEN:-365}}
echo "Using seq_len: $PRED_LEN (env=${DIFFUSION_PRED_LEN:-unset}, detected=${DETECTED_SEQ_LEN:-?})"

# This is the key difference from run_camels_mask.sh
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 "$PYTHON_SCRIPT" \
    --dataset_type="CAMELS" \
    --npz_path="$NPZ_PATH" \
    --device="cuda" \
    --batch_size=$BATCH_SIZE \
    --horizon=1 \
    --pred_len=$PRED_LEN \
    --windows=$WINDOWS \
    --load_pretrain=False \
    --epochs=200 \
    --patience=20 \
    --lr=0.001 \
    runs --seeds='[1]'

# Step 3: Evaluate model
echo ""
echo "[Step 3/3] Evaluating predictions..."
echo "======================================"

cd "$DATA_DIR"

# metrics log stored in project output directory
METRICS_LOG="$PROJECT_DIR/output/basin_metrics_log.csv"

# Evaluate with --target_basin to get metrics for the original masked basins
python postprocess_perseg_aligntime.py \
    --pred_dir "$PROJECT_DIR/output/pred" \
    --model_name "${MODEL_NAME}_Stage2" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

python postprocess_perseg_aligntime.py \
    --pred_dir "$PROJECT_DIR/output/pred" \
    --model_name "${MODEL_NAME}_Stage2" \
    --partition tst \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "Stage 2 Complete! (Model: $MODEL_NAME)"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions saved to: $PROJECT_DIR/output/pred/"
echo "Figures saved to: $PROJECT_DIR/output/figure/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
