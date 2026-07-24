#!/bin/bash
# Two-Stage LSTM+Diffusion K-fold Cross-Validation
#
# Stage 1: LSTM (X -> Y) fills prepped.npz with predictions for masked basins
# Stage 2: Diffusion (X + Y_filled -> Y) refines predictions using autoregressive features
#
# First 2 folds reserved for hyperparameter tuning
# This script runs folds 3-N (or specified range)
#
# Usage:
#   bash run_lstm_diffusion_kfold.sh NsDiff              # 22 folds, start from fold 3
#   bash run_lstm_diffusion_kfold.sh TimeGrad 53         # 53 folds, start from fold 3
#   bash run_lstm_diffusion_kfold.sh NsDiff 22 5         # 22 folds, start from fold 5
#   bash run_lstm_diffusion_kfold.sh NsDiff 22 3 12      # 22 folds, run folds 3-12 only
#   bash run_lstm_diffusion_kfold.sh NsDiff 22 13 22     # 22 folds, run folds 13-22 (resume, keep CSV)

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash run_lstm_diffusion_kfold.sh <MODEL_NAME> [NUM_FOLDS] [START_FOLD] [END_FOLD]"
    echo ""
    echo "Two-Stage Pipeline: LSTM (X->Y) + Diffusion (X+Y->Y)"
    echo ""
    echo "Supported diffusion models: NsDiff, TimeGrad, TimeDiff, D3VAE, DiffusionTS, CSDI, CSBI, SSSD"
    echo ""
    echo "Examples:"
    echo "  bash run_lstm_diffusion_kfold.sh NsDiff              # 22 folds, start from fold 3"
    echo "  bash run_lstm_diffusion_kfold.sh TimeGrad 53         # 53 folds, start from fold 3"
    echo "  bash run_lstm_diffusion_kfold.sh NsDiff 22 5         # 22 folds, start from fold 5"
    echo "  bash run_lstm_diffusion_kfold.sh NsDiff 22 3 12      # 22 folds, run folds 3-12 only"
    echo "  bash run_lstm_diffusion_kfold.sh NsDiff 22 13 22     # 22 folds, resume folds 13-22"
    exit 1
fi

MODEL_NAME=$1
NUM_FOLDS=${2:-22}
START_FOLD=${3:-3}  # Default: start from fold 3 (skip folds 1-2 for hyperparameter tuning)
END_FOLD=${4:-$NUM_FOLDS}  # Default: run to the last fold
NUM_HYPERPARAM_FOLDS=2  # Folds 1-2 for hyperparameter tuning

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LSTM_DIR="$SCRIPT_DIR/lstm"
DIFFUSION_DIR="$SCRIPT_DIR/diffusion"
DIFFUSION_SCRIPTS_DIR="$DIFFUSION_DIR/scripts/CAMELS"
DATA_DIR="$SCRIPT_DIR/data_processing"
TEMPORAL_DIR="$(dirname $SCRIPT_DIR)"
CSV_FILE="$TEMPORAL_DIR/denormalized_camels_data_time.parquet"

# Check if model script exists
PYTHON_SCRIPT="$DIFFUSION_DIR/src/experiments/${MODEL_NAME}_CAMELS.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_CAMELS.py first."
    exit 1
fi

# Create output directories
mkdir -p "$LSTM_DIR/output"
mkdir -p "$DIFFUSION_DIR/output"

# Reset CSV files only when starting from fold 3 (first run)
# When resuming (START_FOLD > 3), keep existing CSV files to append results
if [ $START_FOLD -eq 3 ]; then
    echo "First run detected (START_FOLD=3). Resetting CSV log files..."
    # LSTM output
    rm -f "$LSTM_DIR/output/basin_metrics_log.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_tst.csv"
    rm -f "$LSTM_DIR/output/vae_statistics_log.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_combined.csv"
    # Diffusion output
    rm -f "$DIFFUSION_DIR/output/basin_metrics_log.csv"
    rm -f "$DIFFUSION_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$DIFFUSION_DIR/output/basin_metrics_log_tst.csv"
    echo "CSV files reset complete."
else
    echo "Resume mode detected (START_FOLD=$START_FOLD). Keeping existing CSV files."
fi
echo ""

echo "=========================================="
echo "LSTM + ${MODEL_NAME} Two-Stage K-Fold CV"
echo "=========================================="
echo "Diffusion Model: $MODEL_NAME"
echo "Total folds: $NUM_FOLDS"
echo "Running folds: $START_FOLD to $END_FOLD"
echo "Hyperparam folds (skipped): 1-$NUM_HYPERPARAM_FOLDS"
echo ""
echo "Pipeline per fold:"
echo "  Stage 1: LSTM (X -> Y) fills masked basin predictions"
echo "  Stage 2: ${MODEL_NAME} (X + Y_filled -> Y) refines"
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

# Calculate start index (skip folds before START_FOLD)
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
    echo "Running Fold $fold/$NUM_FOLDS"
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
    # Stage 1: LSTM
    # ==========================================
    echo "------------------------------------------"
    echo "Stage 1: LSTM Training & NPZ Fill"
    echo "------------------------------------------"
    bash "$LSTM_DIR/run_camels_perstd_stage1.sh" $FOLD_BASINS

    # ==========================================
    # Stage 2: Diffusion
    # ==========================================
    echo ""
    echo "------------------------------------------"
    echo "Stage 2: ${MODEL_NAME} Diffusion Refinement"
    echo "------------------------------------------"
    bash "$DIFFUSION_SCRIPTS_DIR/run_camels_stage2.sh" "$MODEL_NAME" $FOLD_BASINS

    echo ""
    echo "Fold $fold/$NUM_FOLDS completed!"

    START_IDX=$END_IDX
done

echo ""
echo "=========================================="
echo "LSTM + ${MODEL_NAME} Folds $START_FOLD-$END_FOLD Complete!"
echo "=========================================="
echo "Results saved in:"
echo "  LSTM metrics: $LSTM_DIR/output/"
echo "  Diffusion metrics: $DIFFUSION_DIR/output/"
echo "  Predictions: $DIFFUSION_DIR/output/pred/"
echo "  Figures: $DIFFUSION_DIR/output/figure/"
echo ""
if [ $END_FOLD -lt $NUM_FOLDS ]; then
    echo "To continue from fold $((END_FOLD+1)):"
    echo "  bash run.sh $MODEL_NAME $NUM_FOLDS $((END_FOLD+1)) $NUM_FOLDS"
fi
echo "=========================================="
