#!/bin/bash
# K-fold cross-validation for MIDM spatial extrapolation baseline
#
# Mirrors the structure of run_np_kfold.sh:
# divides basins into folds, trains MIDM per fold, evaluates.
#
# Usage:
#   bash run_midm_kfold.sh              # 22 folds, start from fold 3
#   bash run_midm_kfold.sh 22 3 22      # 22 folds, folds 3-22
#   bash run_midm_kfold.sh 22 3 5       # folds 3-5 only

NUM_FOLDS=${1:-22}
START_FOLD=${2:-3}
END_FOLD=${3:-$NUM_FOLDS}
NUM_HYPERPARAM_FOLDS=2

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MIDM_SCRIPTS="$SCRIPT_DIR/scripts"
DATA_DIR="$SCRIPT_DIR/../data_processing"
TEMPORAL_DIR="$(dirname $(dirname $SCRIPT_DIR))"
CSV_FILE="$TEMPORAL_DIR/denormalized_camels_data_time.parquet"

# Reset output directory
echo "Cleaning output directory ..."
rm -rf "$SCRIPT_DIR/output"
mkdir -p "$SCRIPT_DIR/output/pred"

echo "=========================================="
echo "MIDM K-Fold CV (Spatial Extrapolation)"
echo "=========================================="
echo "  Folds: $START_FOLD - $END_FOLD (of $NUM_FOLDS)"
echo "  Hyperparam folds skipped: 1-$NUM_HYPERPARAM_FOLDS"
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

    # --- Preprocess (NPZ must already exist) ---
    NPZ_PATH="$DATA_DIR/data/prepped.npz"
    if [ ! -f "$NPZ_PATH" ]; then
        echo "ERROR: $NPZ_PATH not found."
        echo "Run preprocessing first (e.g. via run_gx_enc.sh for one fold)."
        exit 1
    fi

    # --- Train & Evaluate ---
    bash "$MIDM_SCRIPTS/run_midm_stage.sh" $FOLD_BASINS

    echo "Fold $fold done."
    echo ""
    START_IDX=$END_IDX
done

echo "=========================================="
echo "MIDM K-Fold CV Complete!"
echo "=========================================="
echo "Results: $SCRIPT_DIR/output/"
echo "=========================================="
