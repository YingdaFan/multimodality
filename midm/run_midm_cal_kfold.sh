#!/bin/bash
# K-fold cross-validation: LSTM + MIDM Calibration (SDEdit)
#
# Two-stage pipeline:
#   Stage 1: LSTM (preprocess → mask → VAE → LSTM → fill NPZ)
#   Stage 2: MIDM calibration (noise y_true, condition on y_lstm, SDEdit sampling)
#
# Usage:
#   bash run_midm_cal_kfold.sh              # 22 folds, start from fold 3
#   bash run_midm_cal_kfold.sh 22 3 22      # 22 folds, folds 3-22
#   bash run_midm_cal_kfold.sh 22 3 3       # fold 3 only

NUM_FOLDS=${1:-22}
START_FOLD=${2:-3}
END_FOLD=${3:-$NUM_FOLDS}
NUM_HYPERPARAM_FOLDS=2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MIDM_SCRIPTS="$SCRIPT_DIR/scripts"
IMPUTATION_DIR="$(dirname $SCRIPT_DIR)"
LSTM_DIR="$IMPUTATION_DIR/lstm"
TEMPORAL_DIR="$(dirname $IMPUTATION_DIR)"
CSV_FILE="$TEMPORAL_DIR/denormalized_camels_data_time.parquet"

# Reset output directories
if [ $START_FOLD -eq 3 ]; then
    echo "Resetting output directories ..."
    rm -rf "$SCRIPT_DIR/output"
    mkdir -p "$SCRIPT_DIR/output/pred"
    rm -f "$LSTM_DIR/output/basin_metrics_log.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_trn.csv"
    rm -f "$LSTM_DIR/output/basin_metrics_log_tst.csv"
    rm -f "$LSTM_DIR/output/vae_statistics_log.csv"
fi

echo "=========================================="
echo "LSTM + MIDM Calibration K-Fold CV"
echo "=========================================="
echo "  Folds: $START_FOLD - $END_FOLD (of $NUM_FOLDS)"
echo "=========================================="

# --- Extract unique basin IDs ---
ALL_BASINS=$(python3 -c "\
try:
    import pandas as pd; \
    df = pd.read_parquet('$CSV_FILE', columns=['basin_id']); \
    print(' '.join(sorted(df['basin_id'].unique())))
except Exception:
    import pyarrow.parquet as pq; \
    t = pq.read_table('$CSV_FILE', columns=['basin_id']); \
    print(' '.join(sorted(set(t.column('basin_id').to_pylist()))))")
BASIN_ARRAY=($ALL_BASINS)
TOTAL_BASINS=${#BASIN_ARRAY[@]}

BASINS_PER_FOLD=$((TOTAL_BASINS / NUM_FOLDS))
REMAINDER=$((TOTAL_BASINS % NUM_FOLDS))
echo "Total basins: $TOTAL_BASINS  |  ~$BASINS_PER_FOLD per fold"
echo ""

# --- Compute starting index ---
START_IDX=0
for fold in $(seq 1 $((START_FOLD - 1))); do
    if [ $fold -le $REMAINDER ]; then
        START_IDX=$((START_IDX + BASINS_PER_FOLD + 1))
    else
        START_IDX=$((START_IDX + BASINS_PER_FOLD))
    fi
done

# --- Main loop ---
for fold in $(seq $START_FOLD $END_FOLD); do
    echo "=========================================="
    echo "Fold $fold / $NUM_FOLDS"
    echo "=========================================="

    if [ $fold -le $REMAINDER ]; then
        CURRENT_FOLD_SIZE=$((BASINS_PER_FOLD + 1))
    else
        CURRENT_FOLD_SIZE=$BASINS_PER_FOLD
    fi
    END_IDX=$((START_IDX + CURRENT_FOLD_SIZE))
    FOLD_BASINS="${BASIN_ARRAY[@]:$START_IDX:$CURRENT_FOLD_SIZE}"

    echo "Target basins ($CURRENT_FOLD_SIZE): $(echo $FOLD_BASINS | cut -d' ' -f1-5) ..."

    # Run both stages
    bash "$MIDM_SCRIPTS/run_midm_cal_stage.sh" $FOLD_BASINS

    echo "Fold $fold done."
    echo ""
    START_IDX=$END_IDX
done

echo "=========================================="
echo "LSTM + MIDM Calibration K-Fold CV Complete!"
echo "=========================================="
echo "LSTM metrics: $LSTM_DIR/output/"
echo "MIDM Cal metrics: $SCRIPT_DIR/output/"
echo "=========================================="
