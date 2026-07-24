#!/bin/bash
# Two-Stage LSTM+StochasticInterpolant K-fold Cross-Validation - Pure Encoder Pipeline
#
# Daily CAMELS variant. Parallel to run_gx_enc.sh (diffcal) but Stage 2 uses
# the Stochastic Interpolant framework (sical_gx_enc.py), which provides a
# unified ODE/SDE family. fmcal is recovered as the SI_SIGMA_INT=0,
# SI_EPS_INFERENCE=0 special case (sanity check).
#
# Outputs are written to diffusion/output_si/ to avoid colliding with
# diffusion/output/ (diffcal) and diffusion/output_fm/ (fmcal). LSTM CSVs at
# lstm/output/ are still shared across diffcal/fmcal/sical pipelines -- if you
# alternate between SI, FM and diffusion runs on the same folds, expect those
# CSVs to be overwritten.
#
# SI branch is selected at runtime via env vars:
#   SI_SIGMA_INT=0      SI_EPS_INFERENCE=0      -> fmcal-equivalent (sanity)
#   SI_SIGMA_INT>0      SI_EPS_INFERENCE=0      -> SI ODE branch
#   SI_SIGMA_INT>0      SI_EPS_INFERENCE>0      -> SI SDE branch (diffcal-style)
#
# Env vars (SI-specific):
#   SI_SIGMA_INT       - interpolant noise scale (default: 0.3)
#   SI_EPS_INFERENCE   - SDE diffusion coeff (default: 0.0 = ODE)
#   SI_LAMBDA_SCORE    - eta loss weight (default: 1.0)
# Env vars (shared with fmcal):
#   FM_SOURCE_SIGMA    - source noise scale (default: -1 = auto-estimate)
#   FM_STEPS           - inference Euler steps (default: 20)
#
# Usage:
#   bash run_si_gx_enc.sh sical              # 22 folds, start from fold 3
#   bash run_si_gx_enc.sh sical 22 3 22      # 22 folds, run folds 3-22
#   bash run_si_gx_enc.sh scatter sical 22 3 22  # With scatter fusion

if [ $# -lt 1 ]; then
    echo "Usage: bash run_si_gx_enc.sh [FUSION_TYPE] <MODEL_NAME> [NUM_FOLDS] [START_FOLD] [END_FOLD]"
    echo ""
    echo "Two-Stage Pure Encoder SI Pipeline: LSTM (X->Y) + StochasticInterpolant"
    echo ""
    echo "Key features:"
    echo "  - Stage 1 identical to run_gx_enc.sh (LSTM training + RAW NPZ fill)"
    echo "  - Stage 2 uses Stochastic Interpolant: unified ODE/SDE family"
    echo "  - Outputs isolated to diffusion/output_si/"
    echo ""
    echo "Supported models: sical (uses _gx_enc SI version)"
    echo "Optional FUSION_TYPE: scatter, interference, scatterinterference"
    echo ""
    echo "Examples:"
    echo "  bash run_si_gx_enc.sh sical                                          # default 22 folds"
    echo "  bash run_si_gx_enc.sh sical 22 3 22                                  # folds 3-22"
    echo "  SI_SIGMA_INT=0 SI_EPS_INFERENCE=0 bash run_si_gx_enc.sh sical 22 3 3 # sanity check (fmcal-equivalent)"
    echo "  SI_SIGMA_INT=0.3 SI_EPS_INFERENCE=0 bash run_si_gx_enc.sh sical      # default SI ODE branch"
    echo "  SI_SIGMA_INT=0.3 SI_EPS_INFERENCE=0.5 bash run_si_gx_enc.sh sical    # SI SDE branch"
    exit 1
fi

FUSION_TYPE=""
if [[ "$1" == "interference" || "$1" == "scatter" || "$1" == "scatterinterference" ]]; then
    FUSION_TYPE=$1
    shift
fi

MODEL_NAME=$1
NUM_FOLDS=${2:-22}
START_FOLD=${3:-3}
END_FOLD=${4:-$NUM_FOLDS}
NUM_HYPERPARAM_FOLDS=2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LSTM_DIR="$SCRIPT_DIR/lstm"
DIFFUSION_DIR="$SCRIPT_DIR/diffusion"
DIFFUSION_SCRIPTS_DIR="$DIFFUSION_DIR/scripts/CAMELS"
DATA_DIR="$SCRIPT_DIR/data_processing"
TEMPORAL_DIR="$(dirname $SCRIPT_DIR)"
CSV_FILE="${CSV_FILE:-$TEMPORAL_DIR/camelsh_global_daily.parquet}"

# SI output dir (isolated from diffcal's output/ and fmcal's output_fm/)
SI_OUTPUT_DIR="$DIFFUSION_DIR/output_si"

PYTHON_SCRIPT="$DIFFUSION_DIR/src/experiments/${MODEL_NAME}_gx_enc.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: SI model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_gx_enc.py first."
    exit 1
fi

mkdir -p "$LSTM_DIR/output"
mkdir -p "$SI_OUTPUT_DIR"

# Reset SI CSV files only when starting from fold 3.
# Note: LSTM CSVs are SHARED across diffcal/fmcal/sical pipelines; we still
# reset them at fold 3 since each fold's Stage 1 retrains LSTM from scratch.
if [ $START_FOLD -eq 3 ]; then
    echo "First run detected (START_FOLD=3). Resetting SI CSV log files..."
    rm -f "$LSTM_DIR/output/basin_metrics_log.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_tst.csv"
    rm -f "$LSTM_DIR/output/vae_statistics_log.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_combined.csv"
    rm -f "$SI_OUTPUT_DIR/basin_metrics_log.csv"
    rm -f "$SI_OUTPUT_DIR/basin_metrics_log_trn.csv"
    rm -f "$SI_OUTPUT_DIR/basin_metrics_log_tst.csv"
    echo "SI CSV files reset complete."
else
    echo "Resume mode detected (START_FOLD=$START_FOLD). Keeping existing CSV files."
fi
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
echo "LSTM + ${MODEL_NAME}_gx_enc (StochasticInterpolant) Two-Stage K-Fold CV"
echo "=========================================="
if [ -n "$FUSION_TYPE" ]; then
    echo "SI Model: ${MODEL_NAME}_gx_enc (Pure Encoder + ${FUSION_TYPE} fusion)"
else
    echo "SI Model: ${MODEL_NAME}_gx_enc (Pure Encoder)"
fi
echo "Total folds: $NUM_FOLDS"
echo "Running folds: $START_FOLD to $END_FOLD"
echo "Hyperparam folds (skipped): 1-$NUM_HYPERPARAM_FOLDS"
echo ""
echo "STOCHASTIC INTERPOLANT PIPELINE CONFIGURATION:"
echo "  - Stage 2 uses Pure Encoder backbone (FMDiff, NsDiff variant)"
echo "  - Stage 2 trains via SI loss: ||b - b_target||^2 + lambda * ||eta - z||^2 + recon"
echo "  - Source sigma:    ${FM_SOURCE_SIGMA:-auto}"
echo "  - Inference steps: ${FM_STEPS:-20}"
echo "  - sigma_int:       ${SI_SIGMA_INT:-0.3}"
echo "  - eps_inference:   ${SI_EPS_INFERENCE:-0.0}"
echo "  - lambda_score:    ${SI_LAMBDA_SCORE:-1.0}"
echo "  - Active branch:   $SI_BRANCH"
echo "  - Output dir:      $SI_OUTPUT_DIR (isolated from output/ and output_fm/)"
if [ -n "$FUSION_TYPE" ]; then
    echo "  - Wave Fusion:     ${FUSION_TYPE} (bidirectional)"
else
    echo "  - Wave Fusion:     disabled (unidirectional)"
fi
echo ""
echo "Pipeline per fold:"
echo "  Stage 1: LSTM (X -> Y) + denormalize + fill y_obs_*"
echo "  Stage 2: ${MODEL_NAME}_gx_enc Stochastic Interpolant (X + Y_obs -> Y_raw, Pure Encoder)"
echo "=========================================="

echo "Extracting unique basin IDs..."
ALL_BASINS=$(python3 -c "\
import pandas as pd; \
df = pd.read_parquet('$CSV_FILE', columns=['basin_id']); \
print(' '.join(sorted(df['basin_id'].unique())))")

BASIN_ARRAY=($ALL_BASINS)
TOTAL_BASINS=${#BASIN_ARRAY[@]}

echo "Total basins found: $TOTAL_BASINS"
echo "Number of folds: $NUM_FOLDS"
echo ""

if [ $NUM_FOLDS -gt $TOTAL_BASINS ]; then
    echo "ERROR: NUM_FOLDS ($NUM_FOLDS) cannot be greater than TOTAL_BASINS ($TOTAL_BASINS)"
    exit 1
fi

BASINS_PER_FOLD=$((TOTAL_BASINS / NUM_FOLDS))
REMAINDER=$((TOTAL_BASINS % NUM_FOLDS))

echo "Basins per fold: ~$BASINS_PER_FOLD"
if [ $REMAINDER -gt 0 ]; then
    echo "Note: First $REMAINDER folds will have $((BASINS_PER_FOLD + 1)) basins"
fi
echo ""

START_IDX=0
for fold in $(seq 1 $((START_FOLD - 1))); do
    if [ $fold -le $REMAINDER ]; then
        START_IDX=$((START_IDX + BASINS_PER_FOLD + 1))
    else
        START_IDX=$((START_IDX + BASINS_PER_FOLD))
    fi
done

for fold in $(seq $START_FOLD $END_FOLD); do
    echo ""
    echo "=========================================="
    echo "Running Fold $fold/$NUM_FOLDS (Pure Encoder SI Pipeline)"
    echo "=========================================="

    if [ $fold -le $REMAINDER ]; then
        CURRENT_FOLD_SIZE=$((BASINS_PER_FOLD + 1))
    else
        CURRENT_FOLD_SIZE=$BASINS_PER_FOLD
    fi

    END_IDX=$((START_IDX + CURRENT_FOLD_SIZE))

    FOLD_BASINS="${BASIN_ARRAY[@]:$START_IDX:$CURRENT_FOLD_SIZE}"

    echo "Processing $CURRENT_FOLD_SIZE target basin(s) (indices $START_IDX to $((END_IDX-1)))"
    if [ $CURRENT_FOLD_SIZE -le 5 ]; then
        echo "Target basin(s): $FOLD_BASINS"
    else
        echo "First 5 basins: $(echo $FOLD_BASINS | cut -d' ' -f1-5)"
    fi
    echo ""

    # ==========================================
    # Stage 1: LSTM (RAW version) -- shared with diffcal/fmcal pipelines
    # ==========================================
    echo "------------------------------------------"
    echo "Stage 1: LSTM Training & RAW NPZ Fill"
    echo "------------------------------------------"
    bash "$LSTM_DIR/run_camels_perstd_stage1_raw.sh" $FOLD_BASINS

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
    bash "$DIFFUSION_SCRIPTS_DIR/run_si_gx_enc_stage2.sh" "$MODEL_NAME" $FUSION_TYPE $FOLD_BASINS

    echo ""
    echo "Fold $fold/$NUM_FOLDS completed!"

    START_IDX=$END_IDX
done

echo ""
echo "=========================================="
echo "LSTM + ${MODEL_NAME}_gx_enc (StochasticInterpolant) Folds $START_FOLD-$END_FOLD Complete!"
echo "=========================================="
echo "Pure Encoder SI Pipeline: Stage 2 used Stochastic Interpolant ($SI_BRANCH)"
echo "Results saved in:"
echo "  LSTM metrics: $LSTM_DIR/output/   (shared with diffcal/fmcal pipelines)"
echo "  SI metrics:   $SI_OUTPUT_DIR/"
echo "  Predictions:  $SI_OUTPUT_DIR/pred/"
echo "  Figures:      $SI_OUTPUT_DIR/figure/"
echo ""
if [ $END_FOLD -lt $NUM_FOLDS ]; then
    echo "To continue from fold $((END_FOLD+1)):"
    echo "  bash run_si_gx_enc.sh $MODEL_NAME $NUM_FOLDS $((END_FOLD+1)) $NUM_FOLDS"
fi
echo "=========================================="
