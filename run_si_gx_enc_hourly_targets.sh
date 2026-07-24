#!/bin/bash
# Single-run SI calibration on a user-specified set of target basins (hourly).
#
# Replaces the k-fold loop in run_si_gx_enc_hourly.sh with one explicit call:
# you pick the basins (e.g. via data_processing/analyze_basin_completeness.py)
# and pass them in. Useful when you want to evaluate on a curated low-NaN
# subset rather than do 10-fold CV across all 618 basins.
#
# How target basins are passed:
#   - via TARGET_BASINS env var (space-separated string), OR
#   - as positional arguments after the model name
#
# Env vars:
#   TARGET_BASINS      - space-separated basin IDs (cleanest-first recommended)
#   SI_SIGMA_INT       - interpolant noise scale (default: 0.3)
#   SI_EPS_INFERENCE   - SDE diffusion coeff (default: 0.0 = ODE)
#   SI_LAMBDA_SCORE    - eta loss weight (default: 1.0)
#   FM_SOURCE_SIGMA    - source noise scale (default: -1 = auto-estimate)
#   FM_STEPS           - inference Euler steps (default: 20)
#
# Usage:
#   # Identify cleanest basins:
#   python data_processing/analyze_basin_completeness.py --top 20
#
#   # Run SI on the top 5 cleanest basins (positional form):
#   bash run_si_gx_enc_hourly_targets.sh sical 03298500 03326500 ...
#
#   # Or via TARGET_BASINS env var:
#   TARGET_BASINS="03298500 03326500 03524000" \
#       bash run_si_gx_enc_hourly_targets.sh sical
#
#   # With SI branch selection (e.g. SDE):
#   SI_SIGMA_INT=0.3 SI_EPS_INFERENCE=0.5 \
#       TARGET_BASINS="03298500 03326500" \
#       bash run_si_gx_enc_hourly_targets.sh sical

if [ $# -lt 1 ]; then
    echo "Usage: bash run_si_gx_enc_hourly_targets.sh [FUSION_TYPE] <MODEL_NAME> [<basin_id> ...]"
    echo ""
    echo "Single-run SI calibration on user-specified target basins (hourly)."
    echo "Replaces the k-fold loop with one explicit call on the chosen basins."
    echo ""
    echo "Specify target basins either via TARGET_BASINS env var or as positional args."
    echo ""
    echo "Examples:"
    echo "  python data_processing/analyze_basin_completeness.py --top 20"
    echo "  TARGET_BASINS=\"03298500 03326500 03524000\" bash $0 sical"
    echo "  bash $0 sical 03298500 03326500"
    echo "  SI_SIGMA_INT=0 SI_EPS_INFERENCE=0 TARGET_BASINS=\"03298500\" bash $0 sical"
    exit 1
fi

FUSION_TYPE=""
if [[ "$1" == "interference" || "$1" == "scatter" || "$1" == "scatterinterference" ]]; then
    FUSION_TYPE=$1
    shift
fi

MODEL_NAME=$1
shift

# Resolve target basins: positional args take precedence, else TARGET_BASINS env var
if [ $# -ge 1 ]; then
    TARGET_LIST=("$@")
elif [ -n "$TARGET_BASINS" ]; then
    read -r -a TARGET_LIST <<< "$TARGET_BASINS"
else
    echo "ERROR: no target basins specified."
    echo "Pass them either as positional args or via TARGET_BASINS env var."
    echo "Run: python data_processing/analyze_basin_completeness.py --top 20"
    exit 1
fi

if [ ${#TARGET_LIST[@]} -eq 0 ]; then
    echo "ERROR: target basin list is empty after parsing."
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LSTM_DIR="$SCRIPT_DIR/lstm"
DIFFUSION_DIR="$SCRIPT_DIR/diffusion"
DIFFUSION_SCRIPTS_DIR="$DIFFUSION_DIR/scripts/CAMELS"
DATA_DIR="$SCRIPT_DIR/data_processing"

# Hourly-specific configuration (matches run_si_gx_enc_hourly.sh)
export PREPROCESS_SCRIPT="preprocess_perseg_aligntime_camelsh.py"
export DIFFUSION_PRED_LEN=168
export DIFFUSION_WINDOWS=168

SI_OUTPUT_DIR="$DIFFUSION_DIR/output_si"
mkdir -p "$LSTM_DIR/output"
mkdir -p "$SI_OUTPUT_DIR"

PYTHON_SCRIPT="$DIFFUSION_DIR/src/experiments/${MODEL_NAME}_gx_enc.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: SI model script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Reset only the target-basin metric file (single-run, no fold accumulation)
echo "Resetting CSV log files for single-run targeted evaluation..."
rm -f "$LSTM_DIR/output/basin_metrics_log.csv"
rm -f "$LSTM_DIR/output/basin_metrics_log_trn.csv"
rm -f "$LSTM_DIR/output/basin_metrics_log_tst.csv"
rm -f "$LSTM_DIR/output/vae_statistics_log.csv"
rm -f "$LSTM_DIR/output/basin_metrics_combined.csv"
rm -f "$SI_OUTPUT_DIR/basin_metrics_log.csv"
rm -f "$SI_OUTPUT_DIR/basin_metrics_log_trn.csv"
rm -f "$SI_OUTPUT_DIR/basin_metrics_log_tst.csv"
echo ""

# Determine active SI branch for logging
SI_INT_VAL=${SI_SIGMA_INT:-0.3}
SI_EPS_VAL=${SI_EPS_INFERENCE:-0.0}
if (( $(echo "$SI_INT_VAL <= 0" | bc -l) )); then
    SI_BRANCH="ODE-only (fmcal-equivalent: sigma_int=0)"
elif (( $(echo "$SI_EPS_VAL == 0" | bc -l) )); then
    SI_BRANCH="SI ODE branch (deterministic)"
else
    SI_BRANCH="SI SDE branch (eps=${SI_EPS_VAL})"
fi

echo "=========================================="
echo "LSTM + ${MODEL_NAME}_gx_enc (StochasticInterpolant) - Targeted Single Run (HOURLY)"
echo "=========================================="
if [ -n "$FUSION_TYPE" ]; then
    echo "SI Model:        ${MODEL_NAME}_gx_enc (Pure Encoder + ${FUSION_TYPE} fusion)"
else
    echo "SI Model:        ${MODEL_NAME}_gx_enc (Pure Encoder)"
fi
echo "Mode:            Single-run on user-specified target basins (no k-fold)"
echo "Target count:    ${#TARGET_LIST[@]}"
if [ ${#TARGET_LIST[@]} -le 10 ]; then
    echo "Target basins:   ${TARGET_LIST[*]}"
else
    echo "First 10 targets: ${TARGET_LIST[@]:0:10} ..."
fi
echo ""
echo "STOCHASTIC INTERPOLANT CONFIGURATION:"
echo "  - Source sigma:    ${FM_SOURCE_SIGMA:-auto}"
echo "  - Inference steps: ${FM_STEPS:-20}"
echo "  - sigma_int:       ${SI_SIGMA_INT:-0.3}"
echo "  - eps_inference:   ${SI_EPS_INFERENCE:-0.0}"
echo "  - lambda_score:    ${SI_LAMBDA_SCORE:-1.0}"
echo "  - Active branch:   $SI_BRANCH"
echo "  - Output dir:      $SI_OUTPUT_DIR"
echo "  - Hourly: pred_len=$DIFFUSION_PRED_LEN, windows=$DIFFUSION_WINDOWS"
echo "=========================================="

# ==========================================
# Stage 1: LSTM (RAW version) -- shared with diffcal/fmcal/sical k-fold scripts
# ==========================================
echo ""
echo "------------------------------------------"
echo "Stage 1: LSTM Training & RAW NPZ Fill"
echo "Targets passed to Stage 1: ${TARGET_LIST[*]}"
echo "------------------------------------------"
bash "$LSTM_DIR/run_camels_perstd_stage1_raw.sh" "${TARGET_LIST[@]}"

# ==========================================
# Stage 2: Stochastic Interpolant (Pure Encoder)
# ==========================================
echo ""
echo "------------------------------------------"
if [ -n "$FUSION_TYPE" ]; then
    echo "Stage 2: ${MODEL_NAME}_gx_enc StochasticInterpolant (Pure Encoder + ${FUSION_TYPE} fusion)"
else
    echo "Stage 2: ${MODEL_NAME}_gx_enc StochasticInterpolant (Pure Encoder)"
fi
echo "Active branch: $SI_BRANCH"
echo "------------------------------------------"
bash "$DIFFUSION_SCRIPTS_DIR/run_si_gx_enc_stage2.sh" "$MODEL_NAME" $FUSION_TYPE "${TARGET_LIST[@]}"

echo ""
echo "=========================================="
echo "Targeted SI run complete."
echo "=========================================="
echo "Active branch: $SI_BRANCH"
echo "Target basins (${#TARGET_LIST[@]}): ${TARGET_LIST[*]}"
echo "Results saved in:"
echo "  LSTM metrics: $LSTM_DIR/output/"
echo "  SI metrics:   $SI_OUTPUT_DIR/"
echo "  Predictions:  $SI_OUTPUT_DIR/pred/"
echo "=========================================="
