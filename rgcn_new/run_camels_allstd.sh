# run_camels.sh
#!/bin/bash

# 接受命令行参数，默认为"01022500"
if [ $# -eq 0 ]; then
    set -- 01022500
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMPUTATION_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$IMPUTATION_DIR/data_processing"

echo "======================================"
echo "RGCN Historical Data Imputation"
echo "Target Basin(s) to mask: $@"
echo "======================================"

cd "$DATA_DIR"
# Pass the same basin names to preprocessing to exclude them from normalization
echo "Step 1: Preprocessing data (excluding $@ from Y normalization calculation)..."
python preprocess_allbasin_aligntime_camels.py "$@"

echo "Step 2: Masking basin(s) $@ in training data..."
python modify_basin_to_nan_allmask.py "$@"

cd "$SCRIPT_DIR"
python base.py
python evaluate.py --target_basin "$*"

echo "Pipeline complete for basin(s): $@"
