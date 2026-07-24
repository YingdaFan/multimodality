#!/bin/bash
set -e
# Stage 2 of LSTM+Diffusion pipeline using Pure Encoder backbone
#
# Usage: bash run_gx_enc_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>
#   FUSION_TYPE: interference, scatter, scatterinterference (optional)
#
# Example:
#   bash run_gx_enc_stage2.sh diffcal 01022500 02069700              # No wave fusion
#   bash run_gx_enc_stage2.sh diffcal interference 01022500 02069700 # With interference

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash run_gx_enc_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>"
    echo ""
    echo "Supported models: diffcal (uses _gx_enc version with pure Encoder)"
    echo "Optional FUSION_TYPE: interference, scatter, scatterinterference"
    echo ""
    echo "This script is Stage 2 (Pure Encoder) of the LSTM+Diffusion pipeline."
    echo ""
    echo "Examples:"
    echo "  bash run_gx_enc_stage2.sh diffcal 01022500 02069700              # No fusion"
    echo "  bash run_gx_enc_stage2.sh diffcal interference 01022500 02069700 # Interference"
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

# Get target basins
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
echo "Stage 2: Diffusion with Pure Encoder"
echo "======================================"
if [ -n "$FUSION_TYPE" ]; then
    echo "Model: diffusion_gx_enc with ${FUSION_TYPE} fusion"
    echo "Wave Fusion: ENABLED (bidirectional)"
else
    echo "Model: diffusion_gx_enc"
    echo "Wave Fusion: DISABLED (unidirectional)"
fi
echo "Target basins: ${TARGET_BASINS[*]}"
echo ""
echo "NOTE: prepped.npz should have y_raw_* filled by Stage 1"
echo "======================================"

# Skip Step 1: Preprocessing
echo ""
echo "[Step 1/3] Skipping preprocessing (using Stage 1 filled prepped.npz)"
echo "======================================"

# Step 2: Train diffusion model (Pure Encoder version)
echo ""
echo "[Step 2/3] Training ${MODEL_NAME}_gx_enc model..."
echo "======================================"

# Build Python script path (use gx_enc version)
PYTHON_SCRIPT="./src/experiments/${MODEL_NAME}_gx_enc.py"

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_gx_enc.py first."
    exit 1
fi

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

# Train with Pure Encoder backbone
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

# Use RAW-specific postprocess script
python postprocess_perseg_aligntime_raw.py \
    --pred_dir "$PROJECT_DIR/output/pred" \
    --model_name "diffusion_gx_enc" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "Stage 2 (Pure Encoder) Complete! (Model: diffusion_gx_enc)"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions saved to: $PROJECT_DIR/output/pred/"
echo "Figures saved to: $PROJECT_DIR/output/figure/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
