#!/bin/bash
set -e
# Stage 2 of LSTM+Diffusion Joint End-to-End Training (Pure Encoder backbone)
#
# Key difference from run_gx_enc_stage2.sh:
#   - Loads pre-trained LSTM into the diffusion computation graph
#   - Diffusion loss backpropagates through LSTM via differentiable denormalization
#   - LSTM fine-tuned with small LR, diffusion trained at normal LR
#
# Usage: bash run_gx_enc_joint_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>
#
# Example:
#   bash run_gx_enc_joint_stage2.sh diffcal 01022500 02069700
#   bash run_gx_enc_joint_stage2.sh diffcal interference 01022500 02069700

if [ $# -lt 2 ]; then
    echo "Usage: bash run_gx_enc_joint_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>"
    echo ""
    echo "Joint end-to-end LSTM+Diffusion training (Pure Encoder backbone)"
    echo "Uses diffcal_gx_enc_joint.py"
    echo ""
    echo "Env vars:"
    echo "  LSTM_WEIGHTS_PATH  - path to pre-trained LSTM weights (default: ../lstm/output/finetuned_weights.pth)"
    echo "  LSTM_LR            - LSTM learning rate (default: 1e-5)"
    echo "  DIFFUSION_PRED_LEN - prediction length (default: 365)"
    echo "  DIFFUSION_WINDOWS  - window length (default: 365)"
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

# LSTM parameters (configurable via env vars)
LSTM_WEIGHTS=${LSTM_WEIGHTS_PATH:-"../lstm/output/finetuned_weights.pth"}
LSTM_LEARNING_RATE=${LSTM_LR:-1e-5}

echo "======================================"
echo "Stage 2: Joint LSTM+Diffusion (Pure Encoder)"
echo "======================================"
if [ -n "$FUSION_TYPE" ]; then
    echo "Model: diffusion_gx_enc_joint with ${FUSION_TYPE} fusion"
else
    echo "Model: diffusion_gx_enc_joint"
fi
echo "Target basins: ${TARGET_BASINS[*]}"
echo ""
echo "LSTM config:"
echo "  Weights: $LSTM_WEIGHTS"
echo "  Learning rate: $LSTM_LEARNING_RATE"
echo ""
echo "NOTE: prepped.npz should have y_raw_* filled by Stage 1"
echo "======================================"

echo ""
echo "[Step 1/3] Skipping preprocessing (using Stage 1 filled prepped.npz)"
echo "======================================"

echo ""
echo "[Step 2/3] Joint training ${MODEL_NAME}_gx_enc_joint..."
echo "======================================"

PYTHON_SCRIPT="./src/experiments/${MODEL_NAME}_gx_enc_joint_smap.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Joint model script not found: $PYTHON_SCRIPT"
    echo "Expected: ${MODEL_NAME}_gx_enc_joint_smap.py"
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
JOINT_SMAP=1 \
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
    --patience=100 \
    --lr=0.001 \
    --fusion_type="$FUSION_TYPE" \
    --prior_weights_path="$LSTM_WEIGHTS" \
    --prior_lr=$LSTM_LEARNING_RATE \
    --masked_basin_ids ${TARGET_BASINS[@]} \
    runs --seeds='[1]'

echo ""
echo "[Step 3/3] Evaluating predictions..."
echo "======================================"

cd "$DATA_DIR"

METRICS_LOG="$PROJECT_DIR/output/basin_metrics_log.csv"

python postprocess_perseg_aligntime_raw.py \
    --pred_dir "$PROJECT_DIR/output/pred" \
    --model_name "diffusion_gx_enc_joint" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "Stage 2 (Joint) Complete!"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions saved to: $PROJECT_DIR/output/pred/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
