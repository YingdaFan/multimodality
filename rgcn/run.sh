# run.sh
#!/bin/bash

# 接受命令行参数，默认为"CAU BSR"
if [ $# -eq 0 ]; then
    set -- CAU BSR
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMPUTATION_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$IMPUTATION_DIR/data_processing"

echo "======================================"
echo "RGCN Historical Data Imputation"
echo "Target Basin(s): $@"
echo "======================================"

cd "$DATA_DIR"
python preprocess_perseg_aligntime2.py
python modify_basin_to_nan_allmask.py "$@"

cd "$SCRIPT_DIR"
python base.py
python evaluate.py --target_basin "$*"

echo "Pipeline complete for basin(s): $@"
