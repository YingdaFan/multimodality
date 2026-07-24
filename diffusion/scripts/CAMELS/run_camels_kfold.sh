#!/bin/bash
# Generic K-fold cross-validation for CAMELS imputation (with basin mask)
# Supports all diffusion models: NsDiff, TimeGrad, TimeDiff, D3VAE, DiffusionTS, etc.
#
# First 2 folds reserved for hyperparameter tuning
# This script runs folds 3-N (or specified NUM_FOLDS)
#
# Usage:
#   bash run_camels_kfold.sh NsDiff                # NsDiff, default 22 folds
#   bash run_camels_kfold.sh TimeGrad 53           # TimeGrad, specify 53 folds
#   bash run_camels_kfold.sh NsDiff 22 5           # NsDiff, 22 folds, start from fold 5
#   bash run_camels_kfold.sh NsDiff 22 3 12        # NsDiff, 22 folds, run folds 3-12
#   bash run_camels_kfold.sh NsDiff 22 13 22       # NsDiff, 22 folds, run folds 13-22 (resume, keep CSV)

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: bash run_camels_kfold.sh <MODEL_NAME> [NUM_FOLDS] [START_FOLD] [END_FOLD]"
    echo ""
    echo "Supported models: NsDiff, TimeGrad, TimeDiff, D3VAE, DiffusionTS, CSDI, CSBI, SSSD"
    echo ""
    echo "Examples:"
    echo "  bash run_camels_kfold.sh NsDiff              # 22 folds, start from fold 3"
    echo "  bash run_camels_kfold.sh TimeGrad 53         # 53 folds, start from fold 3"
    echo "  bash run_camels_kfold.sh NsDiff 22 5         # 22 folds, start from fold 5"
    echo "  bash run_camels_kfold.sh NsDiff 22 3 12      # 22 folds, run folds 3-12 only"
    echo "  bash run_camels_kfold.sh NsDiff 22 13 22     # 22 folds, run folds 13-22 (resume, keep CSV)"
    exit 1
fi

MODEL_NAME=$1
NUM_FOLDS=${2:-22}
START_FOLD=${3:-3}  # Default: start from fold 3 (skip first 2 folds for hyperparameter tuning)
END_FOLD=${4:-$NUM_FOLDS}  # Default: run until the last fold
NUM_HYPERPARAM_FOLDS=2  # Folds 1-2 for hyperparameter tuning

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname $(dirname $SCRIPT_DIR))"
DATA_DIR="$(dirname $PROJECT_DIR)/data_processing"
# CSV file is in the temporal/ directory (grandparent of imputation)
TEMPORAL_DIR="$(dirname $(dirname $PROJECT_DIR))"
CSV_FILE="$TEMPORAL_DIR/denormalized_camels_data_time.parquet"

# Check if the model script exists
PYTHON_SCRIPT="$PROJECT_DIR/src/experiments/${MODEL_NAME}_CAMELS.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Model script not found: $PYTHON_SCRIPT"
    echo "Please create ${MODEL_NAME}_CAMELS.py first."
    exit 1
fi

# Create output directory
mkdir -p "$PROJECT_DIR/output"

# Reset CSV files only when starting from fold 3 (first run)
# When resuming (START_FOLD > 3), keep existing CSV files to append results
if [ $START_FOLD -eq 3 ]; then
    echo "First run detected (START_FOLD=3). Resetting CSV log files..."
    rm -f "$PROJECT_DIR/output/basin_metrics_log.csv"
    rm -f "$PROJECT_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$PROJECT_DIR/output/basin_metrics_log_tst.csv"
    rm -f "$PROJECT_DIR/output/vae_statistics_log.csv"
    rm -f "$PROJECT_DIR/output/basin_metrics_combined.csv"
    echo "CSV files reset complete."
else
    echo "Resume mode detected (START_FOLD=$START_FOLD). Keeping existing CSV files."
fi
echo ""

echo "=========================================="
echo "${MODEL_NAME} K-Fold Cross-Validation"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Total folds: $NUM_FOLDS"
echo "Running folds: $START_FOLD to $END_FOLD"
echo "Hyperparam folds (skipped): 1-$NUM_HYPERPARAM_FOLDS"
echo "=========================================="

# Extract all unique basin IDs from the raw CSV (preserving leading zeros)
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

# Validate the number of folds
if [ $NUM_FOLDS -gt $TOTAL_BASINS ]; then
    echo "ERROR: NUM_FOLDS ($NUM_FOLDS) cannot be greater than TOTAL_BASINS ($TOTAL_BASINS)"
    exit 1
fi

# Calculate the size of each fold
BASINS_PER_FOLD=$((TOTAL_BASINS / NUM_FOLDS))
REMAINDER=$((TOTAL_BASINS % NUM_FOLDS))

echo "Basins per fold: ~$BASINS_PER_FOLD"
if [ $REMAINDER -gt 0 ]; then
    echo "Note: First $REMAINDER folds will have $((BASINS_PER_FOLD + 1)) basins"
fi
echo ""

# Calculate the starting position (skip the specified folds)
START_IDX=0
for fold in $(seq 1 $((START_FOLD - 1))); do
    if [ $fold -le $REMAINDER ]; then
        START_IDX=$((START_IDX + BASINS_PER_FOLD + 1))
    else
        START_IDX=$((START_IDX + BASINS_PER_FOLD))
    fi
done

# Loop through each fold
for fold in $(seq $START_FOLD $END_FOLD); do
    echo ""
    echo "=========================================="
    echo "Running Fold $fold/$NUM_FOLDS (Model: $MODEL_NAME)"
    echo "=========================================="

    # Calculate the size of the current fold
    if [ $fold -le $REMAINDER ]; then
        CURRENT_FOLD_SIZE=$((BASINS_PER_FOLD + 1))
    else
        CURRENT_FOLD_SIZE=$BASINS_PER_FOLD
    fi

    END_IDX=$((START_IDX + CURRENT_FOLD_SIZE))

    # Extract basin IDs for the current fold
    FOLD_BASINS="${BASIN_ARRAY[@]:$START_IDX:$CURRENT_FOLD_SIZE}"

    echo "Processing $CURRENT_FOLD_SIZE target basin(s) (indices $START_IDX to $((END_IDX-1)))"
    if [ $CURRENT_FOLD_SIZE -le 5 ]; then
        echo "Target basin(s): $FOLD_BASINS"
    else
        echo "First 5 basins: $(echo $FOLD_BASINS | cut -d' ' -f1-5)"
    fi
    echo ""

    # Call the generic run_camels_mask.sh, passing the model name and basin ID list for the current fold
    bash "$SCRIPT_DIR/run_camels_mask.sh" "$MODEL_NAME" $FOLD_BASINS

    echo ""
    echo "Fold $fold/$NUM_FOLDS completed!"

    START_IDX=$END_IDX
done

echo ""
echo "=========================================="
echo "${MODEL_NAME} ${NUM_FOLDS}-Fold Cross-Validation Complete!"
echo "=========================================="
echo "All results saved in: $PROJECT_DIR/output/"
echo "  - basin_metrics_log_trn.csv"
echo "  - basin_metrics_log_tst.csv"
echo "  - pred/ (predictions)"
echo "  - figure/ (visualizations)"
echo "=========================================="
