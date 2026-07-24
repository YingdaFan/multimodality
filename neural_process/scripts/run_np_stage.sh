#!/bin/bash
set -e
# Single-fold NP baseline: train + evaluate
#
# Usage: bash run_np_stage.sh <MODEL_TYPE> <target_basins...>
#   MODEL_TYPE: anp, gnp
#
# Example:
#   bash run_np_stage.sh anp 01022500 02069700
#   bash run_np_stage.sh gnp 01022500 02069700

if [ $# -lt 2 ]; then
    echo "Usage: bash run_np_stage.sh <MODEL_TYPE> <target_basins...>"
    echo "  MODEL_TYPE: anp | gnp"
    exit 1
fi

MODEL_TYPE=$1
shift
TARGET_BASINS=("$@")
TARGET_BASIN="$@"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NP_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$(dirname $NP_DIR)/data_processing"
NPZ_PATH="$DATA_DIR/data/prepped.npz"

echo "======================================"
echo "Neural Process Baseline ($MODEL_TYPE)"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo ""

# --------------------------------------------------
# Step 1: Train NP
# --------------------------------------------------
echo "[Step 1/2] Training ${MODEL_TYPE} ..."
cd "$NP_DIR"

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 train.py \
    --model_type="$MODEL_TYPE" \
    --npz_path="$NPZ_PATH" \
    --device="cuda" \
    --max_context=64 \
    --max_target=32 \
    --hidden_dim=256 \
    --latent_dim=128 \
    --n_heads=4 \
    --epochs=200 \
    --patience=20 \
    --lr=0.001 \
    --beta_kl=0.1 \
    --context_ratio=0.8 \
    --gnn_layers=2 \
    --k_neighbors=10 \
    --masked_basins ${TARGET_BASINS[@]}

# --------------------------------------------------
# Step 2: Evaluate
# --------------------------------------------------
echo ""
echo "[Step 2/2] Evaluating predictions ..."
cd "$DATA_DIR"

METRICS_LOG="$NP_DIR/output/basin_metrics_log.csv"

python3 postprocess_perseg_aligntime_raw.py \
    --pred_dir "$NP_DIR/output/pred" \
    --model_name "neural_process_${MODEL_TYPE}" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "NP Baseline ($MODEL_TYPE) Complete!"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions: $NP_DIR/output/pred/"
echo "Metrics log: $METRICS_LOG"
echo "======================================"
