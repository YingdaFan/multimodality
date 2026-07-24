#!/bin/bash
# Enhanced version of run_camels_perstd.sh with optional validation basin support
#
# Usage 1 (Original): bash run_camels_enhanced.sh <target_basins...>
# Usage 2 (With val): bash run_camels_enhanced.sh --val <val_basins> <test_basins...>

# Accept command line arguments, default to "01022500"
if [ $# -eq 0 ]; then
    set -- 01022500
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMPUTATION_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$IMPUTATION_DIR/data_processing"


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
python preprocess_perseg_aligntime_camels.py




# Mask training set
if [ -n "$TARGET_VAL_BASINS" ]; then
    # Mask both validation and test basins
    python modify_basin_to_nan_allmask.py $TARGET_VAL_BASINS $TARGET_TEST_BASINS
    # Prepare validation set
    python modify_basin_for_validation.py $TARGET_VAL_BASINS
else
    # Only mask test basins
    python modify_basin_to_nan_allmask.py $TARGET_TEST_BASINS
fi

# Use VAE to predict y_mean and y_std for masked basins to avoid information leakage
echo "======================================"
echo "Applying VAE correction for masked basins..."
echo "======================================"
# VAE is only applied to test basins (when validation basins are present)
python apply_vae.py $TARGET_TEST_BASINS --script_dir "$SCRIPT_DIR"

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

# Merge basin metrics with VAE statistics
echo ""
echo "======================================"
echo "Merging basin metrics with VAE statistics..."
echo "======================================"
cd "$DATA_DIR"
python merge_basin_metrics.py --script_dir "$SCRIPT_DIR"

echo "Pipeline complete for basin(s): $TARGET_TEST_BASINS"
