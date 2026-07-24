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
echo "Target Basin(s): $@"
echo "======================================"

cd "$DATA_DIR"
python preprocess_perseg_aligntime_camels.py
python modify_basin_to_nan_allmask.py "$@"

# 使用VAE预测被mask basin的y_mean和y_std，避免信息泄露
echo ""
echo "======================================"
echo "Applying VAE correction for masked basins..."
echo "======================================"
python apply_vae.py "$@" --script_dir "$SCRIPT_DIR"

cd "$SCRIPT_DIR"
python base.py
python evaluate.py --target_basin "$*"

# 合并basin metrics和VAE统计量
echo ""
echo "======================================"
echo "Merging basin metrics with VAE statistics..."
echo "======================================"
cd "$DATA_DIR"
python merge_basin_metrics.py --script_dir "$SCRIPT_DIR"

echo "Pipeline complete for basin(s): $@"
