#!/bin/bash
set -e
# Stage 2 of LSTM+FlowMatching pipeline using Pure Encoder backbone.
# Parallel to run_gx_enc_stage2.sh but invokes fmcal_gx_enc.py and writes to output_fm/.
#
# Usage: bash run_fm_gx_enc_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>
#   MODEL_NAME: typically "fmcal" (resolves to fmcal_gx_enc.py)
#   FUSION_TYPE: interference, scatter, scatterinterference (optional)
#
# Example:
#   bash run_fm_gx_enc_stage2.sh fmcal 01022500 02069700
#   bash run_fm_gx_enc_stage2.sh fmcal interference 01022500 02069700

if [ $# -lt 2 ]; then
    echo "Usage: bash run_fm_gx_enc_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>"
    echo ""
    echo "Supported models: fmcal (uses _gx_enc version with pure Encoder + flow matching)"
    echo "Optional FUSION_TYPE: interference, scatter, scatterinterference"
    echo ""
    echo "This script is Stage 2 (Pure Encoder, Flow Matching) of the LSTM+FM pipeline."
    echo "Outputs go to diffusion/output_fm/ (isolated from diffusion's output/)."
    echo ""
    echo "Examples:"
    echo "  bash run_fm_gx_enc_stage2.sh fmcal 01022500 02069700"
    echo "  bash run_fm_gx_enc_stage2.sh fmcal interference 01022500 02069700"
    exit 1
fi

MODEL_NAME=$1
shift

FUSION_TYPE=""
if [[ "$1" == "interference" || "$1" == "scatter" || "$1" == "scatterinterference" ]]; then
    FUSION_TYPE=$1
    shift
fi

TARGET_BASINS=("$@")
TARGET_BASIN="$@"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
DATA_DIR="$(dirname $PROJECT_DIR)/data_processing"

cd "$PROJECT_DIR"
export PYTHONPATH=./

echo "======================================"
echo "Stage 2: Flow Matching with Pure Encoder"
echo "======================================"
if [ -n "$FUSION_TYPE" ]; then
    echo "Model: flowmatching_gx_enc with ${FUSION_TYPE} fusion"
    echo "Wave Fusion: ENABLED (bidirectional)"
else
    echo "Model: flowmatching_gx_enc"
    echo "Wave Fusion: DISABLED (unidirectional)"
fi
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Output dir: $PROJECT_DIR/output_fm/ (isolated from diffusion's output/)"
echo ""
echo "NOTE: prepped.npz should have y_raw_* filled by Stage 1"
echo "======================================"

echo ""
echo "[Step 1/3] Skipping preprocessing (using Stage 1 filled prepped.npz)"
echo "======================================"

echo ""
echo "[Step 2/3] Training ${MODEL_NAME}_gx_enc model (flow matching)..."
echo "======================================"

PYTHON_SCRIPT="./src/experiments/${MODEL_NAME}_gx_enc.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_gx_enc.py first."
    exit 1
fi

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
    --fm_source_sigma=${FM_SOURCE_SIGMA:--1} \
    --fm_steps=${FM_STEPS:-20} \
    --masked_basin_ids ${TARGET_BASINS[@]} \
    runs --seeds='[1]'

echo ""
echo "[Step 3/3] Evaluating predictions..."
echo "======================================"

cd "$DATA_DIR"

METRICS_LOG="$PROJECT_DIR/output_fm/basin_metrics_log.csv"

python postprocess_perseg_aligntime_raw.py \
    --pred_dir "$PROJECT_DIR/output_fm/pred" \
    --model_name "flowmatching_gx_enc" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "Stage 2 (Pure Encoder, FM) Complete! (Model: flowmatching_gx_enc)"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions saved to: $PROJECT_DIR/output_fm/pred/"
echo "Figures saved to: $PROJECT_DIR/output_fm/figure/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
