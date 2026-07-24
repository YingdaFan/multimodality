#!/bin/bash
# 超参数感知的pipeline脚本
# 用于超参数搜索实验，只运行数据预处理和VAE预测部分
#
# 用法:
#   bash run_camels_perstd_hyperparam.sh --latent_dim 16 --hidden_dim 128 ... <basin_ids>

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMPUTATION_DIR="$(dirname $SCRIPT_DIR)"
DATA_DIR="$IMPUTATION_DIR/data_processing"

# =============================================================================
# 解析命令行参数
# =============================================================================

# 默认超参数值
LATENT_DIM=16
HIDDEN_DIM=128
LR=1e-3
DROPOUT=0.2
EPOCHS=100
BETA_SCHEDULE="linear"
BETA_VALUE=0.5
BATCH_SIZE=32
HYPERPARAM_ID=0
FOLD=0
RESULTS_CSV=""

TARGET_VAL_BASINS=""
TARGET_TEST_BASINS=""

# 解析命名参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --latent_dim)
            LATENT_DIM="$2"
            shift 2
            ;;
        --hidden_dim)
            HIDDEN_DIM="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --beta_schedule)
            BETA_SCHEDULE="$2"
            shift 2
            ;;
        --beta_value)
            BETA_VALUE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --hyperparam_id)
            HYPERPARAM_ID="$2"
            shift 2
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --results_csv)
            RESULTS_CSV="$2"
            shift 2
            ;;
        --val)
            shift
            TARGET_VAL_BASINS="$1"
            shift
            ;;
        *)
            # 剩余的是basin IDs
            TARGET_TEST_BASINS="$@"
            break
            ;;
    esac
done

# 如果没有提供basin IDs，使用默认值
if [ -z "$TARGET_TEST_BASINS" ]; then
    TARGET_TEST_BASINS="01022500"
fi

echo "======================================"
echo "超参数搜索 - 数据Pipeline"
echo "======================================"
echo "超参数组合ID: $HYPERPARAM_ID"
echo "Fold: $FOLD"
echo "超参数配置:"
echo "  latent_dim: $LATENT_DIM"
echo "  hidden_dim: $HIDDEN_DIM"
echo "  lr: $LR"
echo "  dropout: $DROPOUT"
echo "  epochs: $EPOCHS"
echo "  beta_schedule: $BETA_SCHEDULE"
echo "  beta_value: $BETA_VALUE"
echo "  batch_size: $BATCH_SIZE"
echo ""
if [ -n "$TARGET_VAL_BASINS" ]; then
    echo "Validation Basin(s): $TARGET_VAL_BASINS"
    echo "Test Basin(s):       $TARGET_TEST_BASINS"
else
    echo "Target Basin(s): $TARGET_TEST_BASINS"
fi
echo "======================================"

# =============================================================================
# 数据预处理
# =============================================================================

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

# =============================================================================
# VAE预测（使用超参数版本）
# =============================================================================

echo ""
echo "======================================"
echo "应用VAE校正（超参数版本）..."
echo "======================================"

python apply_vae_hyperparam.py $TARGET_TEST_BASINS \
    --script_dir "$SCRIPT_DIR" \
    --latent_dim $LATENT_DIM \
    --hidden_dim $HIDDEN_DIM \
    --lr $LR \
    --dropout $DROPOUT \
    --epochs $EPOCHS \
    --beta_schedule $BETA_SCHEDULE \
    --beta_value $BETA_VALUE \
    --batch_size $BATCH_SIZE \
    --hyperparam_id $HYPERPARAM_ID \
    --fold $FOLD \
    --results_csv "$RESULTS_CSV"



echo "超参数实验完成"
echo "结果已保存到: $RESULTS_CSV"

