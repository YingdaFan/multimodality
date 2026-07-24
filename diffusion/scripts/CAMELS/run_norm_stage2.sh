#!/bin/bash
# Stage 2 of LSTM+Diffusion pipeline (Norm Version - Per-basin Normalized)
# Uses prepped.npz filled by Stage 1 (LSTM predictions for all basins)
#
# Design:
#   - Input: y_obs_* = LSTM predictions (per-basin normalized)
#   - Label: y_raw_* = True observations (per-basin normalized)
#   - Loss masking: masked basins excluded from loss (no information leakage)
#   - Learning: LSTM predictions -> True observations (correction)
#   - Optional fusion_type for bidirectional wave fusion
#
# Usage: bash run_norm_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>
#   FUSION_TYPE: interference, scatter, scatterinterference (optional)
#
# Example:
#   bash run_norm_stage2.sh diffcal 01022500 02069700              # No wave fusion
#   bash run_norm_stage2.sh diffcal interference 01022500 02069700 # With interference

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash run_norm_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>"
    echo ""
    echo "Supported models: diffcal (uses _norm version)"
    echo "Optional FUSION_TYPE: interference, scatter, scatterinterference"
    echo ""
    echo "This script is Stage 2 (Norm) of the LSTM+Diffusion pipeline."
    echo "Key features:"
    echo "  - Input: y_obs_* (LSTM predictions, per-basin normalized)"
    echo "  - Label: y_raw_* (True observations, per-basin normalized)"
    echo "  - Loss masking: masked basins excluded from loss"
    echo "  - Optional bidirectional wave fusion"
    echo ""
    echo "Examples:"
    echo "  bash run_norm_stage2.sh diffcal 01022500 02069700              # No fusion"
    echo "  bash run_norm_stage2.sh diffcal interference 01022500 02069700 # Interference"
    exit 1
fi

# Get model name
MODEL_NAME=$1
shift

# Check if second argument is a fusion_type or a basin ID
FUSION_TYPE=""
if [[ "$1" == "interference" || "$1" == "scatter" || "$1" == "scatterinterference" ]]; then
    FUSION_TYPE=$1
    shift
fi

# Get target basins (for loss masking and evaluation)
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
echo "Stage 2: Diffusion Refinement (Norm)"
echo "======================================"
if [ -n "$FUSION_TYPE" ]; then
    echo "Model: diffusion_norm with ${FUSION_TYPE} fusion"
    echo "Wave Fusion: ENABLED (bidirectional)"
else
    echo "Model: diffusion_norm"
    echo "Wave Fusion: DISABLED (unidirectional)"
fi
echo "Masked basins (loss excluded): ${TARGET_BASINS[*]}"
echo ""
echo "Norm Pipeline Configuration:"
echo "  - Input: y_obs_* (LSTM predictions, per-basin normalized)"
echo "  - Label: y_raw_* (True observations, per-basin normalized)"
echo "  - Loss masking: masked basins excluded from loss"
echo "======================================"

# Skip Step 1: Preprocessing
# We use the prepped.npz that was filled by Stage 1 (LSTM predictions)
echo ""
echo "[Step 1/3] Skipping preprocessing (using Stage 1 filled prepped.npz)"
echo "======================================"

# Step 2: Train diffusion model
echo ""
echo "[Step 2/3] Training ${MODEL_NAME}_norm model..."
echo "======================================"

# Build Python script path (use _norm version)
PYTHON_SCRIPT="./src/experiments/${MODEL_NAME}_norm.py"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_norm.py first."
    exit 1
fi

# Train with --masked_basin_ids for loss masking
# Masked basins: forward pass yes, loss contribution no (avoid information leakage)
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

# fusion_type: empty=unidirectional, interference/scatter/scatterinterference=bidirectional
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
    --fusion_type="$FUSION_TYPE" \
    --masked_basin_ids ${TARGET_BASINS[@]} \
    runs --seeds='[1]'

# Step 3: Evaluate model
echo ""
echo "[Step 3/3] Evaluating predictions..."
echo "======================================"

cd "$DATA_DIR"

# metrics log stored in project output directory
METRICS_LOG="$PROJECT_DIR/output/basin_metrics_log.csv"

# Evaluate with --target_basin to get metrics for the original masked basins
# Use --use_vae_stats to denormalize using y_mean_vae/y_std_vae
python postprocess_perseg_aligntime.py \
    --pred_dir "$PROJECT_DIR/output/pred" \
    --model_name "diffusion_norm" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG" \
    --use_vae_stats

# python postprocess_perseg_aligntime.py \
#     --pred_dir "$PROJECT_DIR/output/pred" \
#     --model_name "diffusion_norm" \
#     --partition tst \
#     --target_basin "$TARGET_BASIN" \
#     --metrics_log "$METRICS_LOG" \
#     --use_vae_stats

echo ""
echo "======================================"
echo "Stage 2 (Norm) Complete! (Model: diffusion_norm)"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions saved to: $PROJECT_DIR/output/pred/"
echo "Figures saved to: $PROJECT_DIR/output/figure/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
