#!/bin/bash
# K-fold: NP predict → fill NPZ → Diffusion calibration
#
# Usage:
#   bash run_np_diff_kfold.sh <MODEL_TYPE> [NUM_FOLDS] [START_FOLD] [END_FOLD]
#   MODEL_TYPE: cnp, tnpd, anp, gnp

if [ $# -lt 1 ]; then
    echo "Usage: bash run_np_diff_kfold.sh <MODEL_TYPE> [NUM_FOLDS] [START_FOLD] [END_FOLD]"
    exit 1
fi

MODEL_TYPE=$1
NUM_FOLDS=${2:-22}
START_FOLD=${3:-3}
END_FOLD=${4:-$NUM_FOLDS}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NP_SCRIPTS="$SCRIPT_DIR/scripts"
TEMPORAL_DIR="$(dirname $(dirname $SCRIPT_DIR))"
CSV_FILE="$TEMPORAL_DIR/denormalized_camels_data_time.parquet"
DATA_DIR="$SCRIPT_DIR/../data_processing"
NPZ_PATH="$DATA_DIR/data/prepped.npz"

# Reset output directory
echo "Cleaning output directory ..."
rm -rf "$SCRIPT_DIR/output"
mkdir -p "$SCRIPT_DIR/output/pred"

echo "=========================================="
echo "NP ($MODEL_TYPE) + Diffusion K-Fold CV"
echo "=========================================="
echo "  Folds: $START_FOLD - $END_FOLD (of $NUM_FOLDS)"
echo "=========================================="

# Check prepped.npz
if [ ! -f "$NPZ_PATH" ]; then
    echo "ERROR: $NPZ_PATH not found. Run preprocessing first."
    exit 1
fi

# Extract basin IDs
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

# Compute starting index
START_IDX=0
for fold in $(seq 1 $((START_FOLD - 1))); do
    if [ $fold -le $REMAINDER ]; then
        START_IDX=$((START_IDX + BASINS_PER_FOLD + 1))
    else
        START_IDX=$((START_IDX + BASINS_PER_FOLD))
    fi
done

# Main loop
for fold in $(seq $START_FOLD $END_FOLD); do
    echo "=========================================="
    echo "Fold $fold / $NUM_FOLDS ($MODEL_TYPE + Diffusion)"
    echo "=========================================="

    if [ $fold -le $REMAINDER ]; then
        CURRENT_FOLD_SIZE=$((BASINS_PER_FOLD + 1))
    else
        CURRENT_FOLD_SIZE=$BASINS_PER_FOLD
    fi
    END_IDX=$((START_IDX + CURRENT_FOLD_SIZE))
    FOLD_BASINS="${BASIN_ARRAY[@]:$START_IDX:$CURRENT_FOLD_SIZE}"

    echo "Target basins ($CURRENT_FOLD_SIZE): $(echo $FOLD_BASINS | cut -d' ' -f1-5) ..."

    bash "$NP_SCRIPTS/run_np_diff_stage.sh" "$MODEL_TYPE" $FOLD_BASINS

    echo "Fold $fold done."
    echo ""
    START_IDX=$END_IDX
done

echo "=========================================="
echo "NP ($MODEL_TYPE) + Diffusion K-Fold Complete!"
echo "=========================================="
echo "Results: $SCRIPT_DIR/output/"
echo "=========================================="
