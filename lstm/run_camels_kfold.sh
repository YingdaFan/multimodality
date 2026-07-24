#!/bin/bash
# K-fold cross-validation for LSTM imputation
# First 2 folds reserved for hyperparameter tuning (run_hyperparam.sh)
#
# Usage:
#   bash run_camels_kfold.sh                    # default 106 folds, from fold 3 to 106
#   bash run_camels_kfold.sh 53                 # 53 folds, from fold 3 to 53
#   bash run_camels_kfold.sh 53 5               # 53 folds, from fold 5 to 53
#   bash run_camels_kfold.sh 53 5 12            # 53 folds, from fold 5 to 12
#   bash run_camels_kfold.sh 53 13 53           # 53 folds, run folds 13-53 (resume, keep CSV)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CSV_FILE="../../denormalized_camels_data_time.parquet"

NUM_FOLDS=${1:-106}
START_FOLD=${2:-3}  # default start from fold 3 (skip first 2 folds for hyperparameter tuning)
END_FOLD=${3:-$NUM_FOLDS}  # default to the last fold
NUM_HYPERPARAM_FOLDS=2  # Folds 1-2 for hyperparameter tuning (~11 basins)

# Reset CSV files only when starting from fold 3 (first run)
# When resuming (START_FOLD > 3), keep existing CSV files to append results
if [ $START_FOLD -eq 3 ]; then
    echo "First run detected (START_FOLD=3). Resetting CSV log files..."
    rm -f "$SCRIPT_DIR/output/basin_metrics_log.csv"
    rm -f "$SCRIPT_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$SCRIPT_DIR/output/basin_metrics_log_tst.csv"
    rm -f "$SCRIPT_DIR/output/vae_statistics_log.csv"
    rm -f "$SCRIPT_DIR/output/basin_metrics_combined.csv"
    echo "CSV files reset complete."
else
    echo "Resume mode detected (START_FOLD=$START_FOLD). Keeping existing CSV files."
fi
echo ""

echo "=========================================="
echo "LSTM ${NUM_FOLDS}-Fold Cross-Validation"
echo "Running folds: $START_FOLD to $END_FOLD"
echo "=========================================="

# Extract all unique basin IDs from CSV (preserving leading zeros)
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

# Calculate fold size
BASINS_PER_FOLD=$((TOTAL_BASINS / NUM_FOLDS))
REMAINDER=$((TOTAL_BASINS % NUM_FOLDS))

echo "Basins per fold: ~$BASINS_PER_FOLD"
if [ $REMAINDER -gt 0 ]; then
    echo "Note: First $REMAINDER folds will have $((BASINS_PER_FOLD + 1)) basins"
fi


echo ""

# Calculate start index (skip specified folds)
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
    echo "=========================================="
    echo "Running Fold $fold/$NUM_FOLDS"
    echo "=========================================="

    # Calculate current fold size (first folds get extra basins from remainder)
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

    # Call run_camels_perstd.sh with basin IDs for current fold
    bash "$SCRIPT_DIR/run_camels_perstd.sh" $FOLD_BASINS

    echo ""
    echo "Fold $fold/$NUM_FOLDS completed!"

    START_IDX=$END_IDX
done

echo "=========================================="
echo "${NUM_FOLDS}-Fold Cross-Validation Complete!"
echo "=========================================="
echo "All results saved in output/basin_metrics_log_trn.csv and output/basin_metrics_log_tst.csv"

