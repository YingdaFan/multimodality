#!/bin/bash
# Enhanced version of run_camels_allstd.sh with optional validation basin support
# Uses global standardization (allbasin) instead of per-basin standardization
#
# Usage 1 (Original): bash run_camels_allstd.sh <target_basins...>
# Usage 2 (With val): bash run_camels_allstd.sh --val <val_basins> <test_basins...>

# Accept command line arguments, default to "01022500"
if [ $# -eq 0 ]; then
    set -- 01022500
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMPUTATION_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$IMPUTATION_DIR/data_processing"

# Check if --val mode is used
TARGET_VAL_BASINS=""
if [ "$1" = "--val" ]; then
    shift
    TARGET_VAL_BASINS="$1"
    shift
fi

TARGET_TEST_BASINS="$@"

echo "======================================"
echo "Historical Data Imputation"
if [ -n "$TARGET_VAL_BASINS" ]; then
    echo "Validation Basin(s): $TARGET_VAL_BASINS"
    echo "Test Basin(s):       $TARGET_TEST_BASINS"
else
    echo "Target Basin(s): $TARGET_TEST_BASINS"
fi
echo "======================================"

cd "$DATA_DIR"

# Pass all masked basins to preprocessing to exclude them from normalization
if [ -n "$TARGET_VAL_BASINS" ]; then
    ALL_BASINS="$TARGET_VAL_BASINS $TARGET_TEST_BASINS"
    echo "Step 1: Preprocessing data (excluding $ALL_BASINS from Y normalization)..."
    python preprocess_allbasin_aligntime_camels.py $ALL_BASINS
else
    echo "Step 1: Preprocessing data (excluding $TARGET_TEST_BASINS from Y normalization)..."
    python preprocess_allbasin_aligntime_camels.py $TARGET_TEST_BASINS
fi

# Mask training set
echo ""
echo "Step 2: Masking basins in training data..."
if [ -n "$TARGET_VAL_BASINS" ]; then
    # Mask both validation and test basins
    python modify_basin_to_nan_allmask.py $TARGET_VAL_BASINS $TARGET_TEST_BASINS
    # Prepare validation set
    python modify_basin_for_validation.py $TARGET_VAL_BASINS
else
    # Only mask test basins
    python modify_basin_to_nan_allmask.py $TARGET_TEST_BASINS
fi

cd "$SCRIPT_DIR"
python base.py

# Evaluate model (call postprocess directly, bypassing evaluate.py wrapper)
# postprocess automatically adds _trn or _tst suffix to filenames
METRICS_LOG="$SCRIPT_DIR/output/basin_metrics_log.csv"
cd "$DATA_DIR"

# Evaluate training set
python postprocess_perseg_aligntime.py \
    --pred_dir "$SCRIPT_DIR/output/preds" \
    --model_name "LSTM" \
    --partition trn \
    --target_basin "$TARGET_TEST_BASINS" \
    --metrics_log "$METRICS_LOG"

# Evaluate test set
python postprocess_perseg_aligntime.py \
    --pred_dir "$SCRIPT_DIR/output/preds" \
    --model_name "LSTM" \
    --partition tst \
    --target_basin "$TARGET_TEST_BASINS" \
    --metrics_log "$METRICS_LOG"

echo "Pipeline complete for basin(s): $TARGET_TEST_BASINS"
