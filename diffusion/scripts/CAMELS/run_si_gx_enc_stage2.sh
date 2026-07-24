#!/bin/bash
set -e
# Stage 2 of LSTM+StochasticInterpolant pipeline using Pure Encoder backbone.
# Parallel to run_fm_gx_enc_stage2.sh; invokes sical_gx_enc.py and writes to
# output_si/.
#
# SI bridges fmcal (ODE branch) and diffcal (SDE branch). Branch is selected
# at runtime via env vars:
#   SI_SIGMA_INT=0   SI_EPS_INFERENCE=0    -> fmcal-equivalent (sanity check)
#   SI_SIGMA_INT>0   SI_EPS_INFERENCE=0    -> SI ODE branch
#   SI_SIGMA_INT>0   SI_EPS_INFERENCE>0    -> SI SDE branch (analog to diffcal)
#
# Usage: bash run_si_gx_enc_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>
#   MODEL_NAME:   typically "sical" (resolves to sical_gx_enc.py)
#   FUSION_TYPE:  interference, scatter, scatterinterference (optional)

if [ $# -lt 2 ]; then
    echo "Usage: bash run_si_gx_enc_stage2.sh <MODEL_NAME> [FUSION_TYPE] <target_basins...>"
    echo ""
    echo "Supported models: sical (uses _gx_enc version with stochastic interpolant)"
    echo "Optional FUSION_TYPE: interference, scatter, scatterinterference"
    echo ""
    echo "Env vars:"
    echo "  FM_SOURCE_SIGMA   - source noise scale (default: -1 = auto-estimate)"
    echo "  FM_STEPS          - inference Euler steps (default: 20)"
    echo "  SI_SIGMA_INT      - interpolant noise scale (default: 0.3)"
    echo "                      0 = fmcal-equivalent"
    echo "  SI_EPS_INFERENCE  - SDE diffusion coeff at inference (default: 0.0)"
    echo "                      0 = ODE branch, >0 = SDE branch"
    echo "  SI_LAMBDA_SCORE   - weight on eta (denoiser) loss (default: 1.0)"
    echo ""
    echo "This is Stage 2 (Pure Encoder, SI) of the LSTM+SI pipeline."
    echo "Outputs go to diffusion/output_si/ (isolated from output/ and output_fm/)."
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
echo "Stage 2: Stochastic Interpolant with Pure Encoder"
echo "======================================"
if [ -n "$FUSION_TYPE" ]; then
    echo "Model: stochastic_interpolant_gx_enc with ${FUSION_TYPE} fusion"
    echo "Wave Fusion: ENABLED (bidirectional)"
else
    echo "Model: stochastic_interpolant_gx_enc"
    echo "Wave Fusion: DISABLED (unidirectional)"
fi
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Output dir: $PROJECT_DIR/output_si/ (isolated from output/ and output_fm/)"
echo ""
echo "SI config:"
echo "  Source sigma:     ${FM_SOURCE_SIGMA:-auto}"
echo "  Inference steps:  ${FM_STEPS:-20}"
echo "  Interpolant noise (sigma_int): ${SI_SIGMA_INT:-0.3}"
echo "  SDE eps_inference:             ${SI_EPS_INFERENCE:-0.0}"
echo "  Score loss weight (lambda):    ${SI_LAMBDA_SCORE:-1.0}"

# Branch summary
SI_INT_VAL=${SI_SIGMA_INT:-0.3}
SI_EPS_VAL=${SI_EPS_INFERENCE:-0.0}
if (( $(echo "$SI_INT_VAL <= 0" | bc -l) )); then
    BRANCH="ODE-only (fmcal-equivalent: sigma_int=0)"
elif (( $(echo "$SI_EPS_VAL == 0" | bc -l) )); then
    BRANCH="SI ODE branch (deterministic)"
else
    BRANCH="SI SDE branch (eps=${SI_EPS_VAL})"
fi
echo "  Active branch:    $BRANCH"
echo ""
echo "NOTE: prepped.npz should have y_raw_* filled by Stage 1"
echo "======================================"

echo ""
echo "[Step 1/3] Skipping preprocessing (using Stage 1 filled prepped.npz)"
echo "======================================"

echo ""
echo "[Step 2/3] Training ${MODEL_NAME}_gx_enc model (stochastic interpolant)..."
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
    --si_sigma_int=${SI_SIGMA_INT:-0.3} \
    --si_eps_inference=${SI_EPS_INFERENCE:-0.0} \
    --si_lambda_score=${SI_LAMBDA_SCORE:-1.0} \
    --masked_basin_ids ${TARGET_BASINS[@]} \
    runs --seeds='[1]'

echo ""
echo "[Step 3/3] Evaluating predictions..."
echo "======================================"

cd "$DATA_DIR"

METRICS_LOG="$PROJECT_DIR/output_si/basin_metrics_log.csv"

python postprocess_perseg_aligntime_raw.py \
    --pred_dir "$PROJECT_DIR/output_si/pred" \
    --model_name "stochastic_interpolant_gx_enc" \
    --partition trn \
    --target_basin "$TARGET_BASIN" \
    --metrics_log "$METRICS_LOG"

echo ""
echo "======================================"
echo "Stage 2 (Pure Encoder, SI) Complete!"
echo "======================================"
echo "Active branch: $BRANCH"
echo "Target basins: ${TARGET_BASINS[*]}"
echo "Predictions saved to: $PROJECT_DIR/output_si/pred/"
echo "Figures saved to:     $PROJECT_DIR/output_si/figure/"
echo "Metrics log:          $METRICS_LOG"
echo "======================================"
