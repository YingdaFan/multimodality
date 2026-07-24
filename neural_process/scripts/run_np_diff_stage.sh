#!/bin/bash
set -e
# Single-fold: NP predict → fill NPZ → Diffusion calibration → Evaluate
#
# Usage: bash run_np_diff_stage.sh <MODEL_TYPE> <target_basins...>
#   MODEL_TYPE: cnp, tnpd, anp, gnp

if [ $# -lt 2 ]; then
    echo "Usage: bash run_np_diff_stage.sh <MODEL_TYPE> <target_basins...>"
    exit 1
fi

MODEL_TYPE=$1
shift
TARGET_BASINS=("$@")
TARGET_BASIN="$@"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NP_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$(dirname $NP_DIR)/data_processing"
DIFFUSION_DIR="$(dirname $NP_DIR)/diffusion"
NPZ_PATH="$DATA_DIR/data/prepped.npz"

echo "======================================"
echo "NP ($MODEL_TYPE) + Diffusion Calibration"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo ""

# --------------------------------------------------
# Step 1/4: NP train + predict
# --------------------------------------------------
echo "[Step 1/5] Training NP ($MODEL_TYPE) ..."
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
# Step 2/5: Evaluate NP-only (stage 1 metrics)
# --------------------------------------------------
echo ""
echo "[Step 2/5] Evaluating NP-only predictions ..."
cd "$DATA_DIR"

NP_METRICS_LOG="$NP_DIR/output/basin_metrics_log_np_only.csv"

python3 postprocess_perseg_aligntime_raw.py \
    --pred_dir "$NP_DIR/output/pred" \
    --model_name "${MODEL_TYPE}_only" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$NP_METRICS_LOG"

# --------------------------------------------------
# Step 3/5: Fill NPZ with NP predictions + stats
# --------------------------------------------------
echo ""
echo "[Step 3/5] Filling NPZ with NP predictions ..."
cd "$NP_DIR"

python3 fill_npz.py \
    --pred_path output/pred/trn.npy \
    --npz_path "$NPZ_PATH"

# --------------------------------------------------
# Step 4/5: Diffusion calibration (existing code, unchanged)
# --------------------------------------------------
echo ""
echo "[Step 4/5] Diffusion calibration ..."
cd "$DIFFUSION_DIR"
export PYTHONPATH=./

BATCH_SIZE=$(python3 -c "\
import numpy as np; \
print(int(np.load('$NPZ_PATH', allow_pickle=True)['n_segs']))")

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 src/experiments/diffcal_gx_enc.py \
    --dataset_type="CAMELS" \
    --npz_path="$NPZ_PATH" \
    --device="cuda" \
    --batch_size=$BATCH_SIZE \
    --horizon=1 \
    --pred_len=${DIFFUSION_PRED_LEN:-365} \
    --windows=${DIFFUSION_WINDOWS:-365} \
    --load_pretrain=False \
    --epochs=200 \
    --patience=20 \
    --lr=0.001 \
    --fusion_type="" \
    --masked_basin_ids ${TARGET_BASINS[@]} \
    runs --seeds='[1]'

# --------------------------------------------------
# Step 4/4: Evaluate diffusion output
# --------------------------------------------------
echo ""
echo "[Step 5/5] Evaluating diffusion output ..."
cd "$DATA_DIR"

METRICS_LOG="$NP_DIR/output/basin_metrics_log.csv"

python3 postprocess_perseg_aligntime_raw.py \
    --pred_dir "$DIFFUSION_DIR/output/pred" \
    --model_name "${MODEL_TYPE}_diffusion" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "NP ($MODEL_TYPE) + Diffusion Complete!"
echo "======================================"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "NP-only metrics:  $NP_METRICS_LOG"
echo "NP+Diff metrics:  $METRICS_LOG"
echo "======================================"
