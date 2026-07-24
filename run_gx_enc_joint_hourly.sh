#!/bin/bash
# Joint LSTM+Diffusion End-to-End K-fold Cross-Validation - Pure Encoder Pipeline (Hourly)
#
# Hourly CAMELS-H variant of run_gx_enc_joint.sh.
# Same as run_gx_enc_hourly.sh but Stage 2 uses joint training:
#   - LSTM is loaded into the diffusion computation graph
#   - Diffusion loss backpropagates into LSTM weights (small LR)
#   - Stage 1 is identical (provides pre-trained LSTM weights + filled NPZ)
#
# Hourly-specific configuration:
#   - Dataset: camelsh_global.parquet (130 TRB + 488 TRB-like = 618 basins)
#   - Preprocess: preprocess_perseg_aligntime_camelsh.py (seq_len=168 = 1 week)
#   - Diffusion: DIFFUSION_PRED_LEN=168, DIFFUSION_WINDOWS=168
#
# Env vars for LSTM joint config:
#   LSTM_LR            - LSTM learning rate (default: 1e-5)
#   LSTM_HIDDEN_DIM    - LSTM hidden dim (default: 20)
#
# Usage:
#   bash run_gx_enc_joint_hourly.sh diffcal              # 10 folds, start from fold 3
#   bash run_gx_enc_joint_hourly.sh diffcal 10 3 10      # 10 folds, run folds 3-10
#   bash run_gx_enc_joint_hourly.sh scatter diffcal 10 3 10  # With scatter fusion

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash run_gx_enc_joint_hourly.sh [FUSION_TYPE] <MODEL_NAME> [NUM_FOLDS] [START_FOLD] [END_FOLD]"
    echo ""
    echo "Joint End-to-End Pipeline (Hourly): LSTM (X->Y) + Diffusion (Joint, Pure Encoder)"
    echo ""
    echo "Data: camelsh_global.parquet (130 TRB + 488 TRB-like = 618 basins, hourly)"
    echo "Preprocessing: preprocess_perseg_aligntime_camelsh.py"
    echo ""
    echo "Key features:"
    echo "  - Stage 1: LSTM pre-training (same as run_gx_enc_hourly.sh)"
    echo "  - Stage 2: Joint LSTM+Diffusion with end-to-end gradient flow"
    echo "  - LSTM fine-tuned with small LR via diffusion loss"
    echo ""
    echo "Supported models: diffcal (uses _gx_enc_joint version)"
    echo "Optional FUSION_TYPE: scatter, interference, scatterinterference"
    echo ""
    echo "Examples:"
    echo "  bash run_gx_enc_joint_hourly.sh diffcal              # Default: 10 folds, no fusion"
    echo "  bash run_gx_enc_joint_hourly.sh diffcal 10 3 10      # Folds 3-10, no fusion"
    echo "  bash run_gx_enc_joint_hourly.sh scatter diffcal 10 3 10  # With scatter fusion"
    echo "  LSTM_LR=5e-6 bash run_gx_enc_joint_hourly.sh diffcal # Custom LSTM LR"
    exit 1
fi

# Check if first argument is a fusion_type or model name
FUSION_TYPE=""
if [[ "$1" == "interference" || "$1" == "scatter" || "$1" == "scatterinterference" ]]; then
    FUSION_TYPE=$1
    shift
fi

MODEL_NAME=$1
NUM_FOLDS=${2:-10}
START_FOLD=${3:-3}  # Default: start from fold 3 (skip folds 1-2 for hyperparameter tuning)
END_FOLD=${4:-$NUM_FOLDS}  # Default: run to the last fold
NUM_HYPERPARAM_FOLDS=2  # Folds 1-2 for hyperparameter tuning

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LSTM_DIR="$SCRIPT_DIR/lstm"
DIFFUSION_DIR="$SCRIPT_DIR/diffusion"
DIFFUSION_SCRIPTS_DIR="$DIFFUSION_DIR/scripts/CAMELS"
DATA_DIR="$SCRIPT_DIR/data_processing"
TEMPORAL_DIR="$(dirname $SCRIPT_DIR)"
CSV_FILE="$TEMPORAL_DIR/camelsh_global.parquet"

# Hourly-specific: tell Stage 1 to use hourly preprocessing
export PREPROCESS_SCRIPT="preprocess_perseg_aligntime_camelsh.py"

# Hourly-specific: diffusion model parameters (seq_len=168 in preprocessing = 1 week)
export DIFFUSION_PRED_LEN=168
export DIFFUSION_WINDOWS=168

# Check if Joint model script exists
PYTHON_SCRIPT="$DIFFUSION_DIR/src/experiments/${MODEL_NAME}_gx_enc_joint.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Joint model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_gx_enc_joint.py first."
    exit 1
fi

# Create output directories
mkdir -p "$LSTM_DIR/output"
mkdir -p "$DIFFUSION_DIR/output"

# Reset CSV files only when starting from fold 3
if [ $START_FOLD -eq 3 ]; then
    echo "First run detected (START_FOLD=3). Resetting CSV log files..."
    rm -f "$LSTM_DIR/output/basin_metrics_log.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_tst.csv"
    rm -f "$LSTM_DIR/output/vae_statistics_log.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_combined.csv"
    rm -f "$DIFFUSION_DIR/output/basin_metrics_log.csv"
    rm -f "$DIFFUSION_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$DIFFUSION_DIR/output/basin_metrics_log_tst.csv"
    echo "CSV files reset complete."
else
    echo "Resume mode detected (START_FOLD=$START_FOLD). Keeping existing CSV files."
fi
echo ""

echo "=========================================="
echo "Joint LSTM+Diffusion K-Fold CV (Pure Encoder, HOURLY)"
echo "=========================================="
echo "Dataset: camelsh_global.parquet (130 TRB + 488 TRB-like = 618 basins, hourly)"
echo "Preprocessing: $PREPROCESS_SCRIPT"
if [ -n "$FUSION_TYPE" ]; then
    echo "Model: ${MODEL_NAME}_gx_enc_joint (Pure Encoder + ${FUSION_TYPE} fusion)"
else
    echo "Model: ${MODEL_NAME}_gx_enc_joint (Pure Encoder)"
fi
echo "Total folds: $NUM_FOLDS"
echo "Running folds: $START_FOLD to $END_FOLD"
echo "Hyperparam folds (skipped): 1-$NUM_HYPERPARAM_FOLDS"
echo "Diffusion pred_len: $DIFFUSION_PRED_LEN, windows: $DIFFUSION_WINDOWS"
echo ""
echo "JOINT PIPELINE CONFIGURATION:"
echo "  - Stage 1: LSTM pre-training (same as run_gx_enc_hourly.sh)"
echo "  - Stage 2: Joint end-to-end (Diffusion loss → LSTM)"
echo "  - LSTM LR: ${LSTM_LR:-1e-5}"
if [ -n "$FUSION_TYPE" ]; then
    echo "  - Wave Fusion: ${FUSION_TYPE} (bidirectional)"
else
    echo "  - Wave Fusion: disabled (unidirectional)"
fi
echo ""
echo "Pipeline per fold:"
echo "  Stage 1: LSTM (X -> Y) + denormalize + fill y_obs_*"
echo "  Stage 2: Joint LSTM+Diffusion (end-to-end, Pure Encoder)"
echo "=========================================="

# Extract unique basin IDs from parquet
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

# Validate fold count
if [ $NUM_FOLDS -gt $TOTAL_BASINS ]; then
    echo "ERROR: NUM_FOLDS ($NUM_FOLDS) cannot be greater than TOTAL_BASINS ($TOTAL_BASINS)"
    exit 1
fi

# Calculate fold sizes
BASINS_PER_FOLD=$((TOTAL_BASINS / NUM_FOLDS))
REMAINDER=$((TOTAL_BASINS % NUM_FOLDS))

echo "Basins per fold: ~$BASINS_PER_FOLD"
if [ $REMAINDER -gt 0 ]; then
    echo "Note: First $REMAINDER folds will have $((BASINS_PER_FOLD + 1)) basins"
fi
echo ""

# Calculate start index
START_IDX=0
for fold in $(seq 1 $((START_FOLD - 1))); do
    if [ $fold -le $REMAINDER ]; then
        START_IDX=$((START_IDX + BASINS_PER_FOLD + 1))
    else
        START_IDX=$((START_IDX + BASINS_PER_FOLD))
    fi
done

# Loop through folds
for fold in $(seq $START_FOLD $END_FOLD); do
    echo ""
    echo "=========================================="
    echo "Running Fold $fold/$NUM_FOLDS (Joint Pipeline - Hourly)"
    echo "=========================================="

    # Calculate current fold size
    if [ $fold -le $REMAINDER ]; then
        CURRENT_FOLD_SIZE=$((BASINS_PER_FOLD + 1))
    else
        CURRENT_FOLD_SIZE=$BASINS_PER_FOLD
    fi

    END_IDX=$((START_IDX + CURRENT_FOLD_SIZE))

    # Extract basin IDs for current fold
    FOLD_BASINS="${BASIN_ARRAY[@]:$START_IDX:$CURRENT_FOLD_SIZE}"

    echo "Processing $CURRENT_FOLD_SIZE target basin(s) (indices $START_IDX to $((END_IDX-1)))"
    if [ $CURRENT_FOLD_SIZE -le 5 ]; then
        echo "Target basin(s): $FOLD_BASINS"
    else
        echo "First 5 basins: $(echo $FOLD_BASINS | cut -d' ' -f1-5)"
    fi
    echo ""

    # ==========================================
    # Stage 1: LSTM (RAW version - same as run_gx_enc_hourly.sh)
    # ==========================================
    echo "------------------------------------------"
    echo "Stage 1: LSTM Training & RAW NPZ Fill"
    echo "------------------------------------------"
    bash "$LSTM_DIR/run_camels_perstd_stage1_raw.sh" $FOLD_BASINS

    # ==========================================
    # Stage 2: Joint LSTM+Diffusion (Pure Encoder)
    # ==========================================
    echo ""
    echo "------------------------------------------"
    if [ -n "$FUSION_TYPE" ]; then
        echo "Stage 2: Joint ${MODEL_NAME}_gx_enc (end-to-end + ${FUSION_TYPE} fusion)"
    else
        echo "Stage 2: Joint ${MODEL_NAME}_gx_enc (end-to-end)"
    fi
    echo "------------------------------------------"
    bash "$DIFFUSION_SCRIPTS_DIR/run_gx_enc_joint_stage2.sh" "$MODEL_NAME" $FUSION_TYPE $FOLD_BASINS

    echo ""
    echo "Fold $fold/$NUM_FOLDS completed!"

    START_IDX=$END_IDX
done

echo ""
echo "=========================================="
echo "Joint LSTM+Diffusion Folds $START_FOLD-$END_FOLD Complete! (HOURLY)"
echo "=========================================="
echo "Joint Pipeline: Stage 2 used end-to-end LSTM+Diffusion"
echo "Results saved in:"
echo "  LSTM metrics: $LSTM_DIR/output/"
echo "  Diffusion metrics: $DIFFUSION_DIR/output/"
echo "  Predictions: $DIFFUSION_DIR/output/pred/"
echo "  Figures: $DIFFUSION_DIR/output/figure/"
echo ""
if [ $END_FOLD -lt $NUM_FOLDS ]; then
    echo "To continue from fold $((END_FOLD+1)):"
    echo "  bash run_gx_enc_joint_hourly.sh $MODEL_NAME $NUM_FOLDS $((END_FOLD+1)) $NUM_FOLDS"
fi
echo "=========================================="
