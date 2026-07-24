#!/bin/bash
set -e
# NsDiff Forecasting Pipeline
#
# 与 imputation 流水线不同，forecasting 是单阶段（无需 LSTM 填充，无需 k-fold）
#
# Usage:
#   bash run_forecast.sh                                    # 默认参数
#   bash run_forecast.sh --windows 168 --pred_len 192       # 自定义窗口
#   bash run_forecast.sh --npz_path /path/to/prepped.npz    # 自定义数据路径
#
# 参数说明:
#   --windows     输入窗口长度（默认 168）
#   --pred_len    预测长度（默认 192）
#   --epochs      训练轮数（默认 200）
#   --patience    Early stopping 耐心值（默认 20）
#   --lr          学习率（默认 0.001）
#   --npz_path    预处理数据路径（默认 ../data_processing/data/prepped.npz）
#   --device      GPU 设备（默认 cuda；用 CUDA_VISIBLE_DEVICES=N 选物理卡）
#   --seeds       随机种子列表（默认 [1]）
#   --step_index  Postprocess 提取第几步预测（默认 0，即 1-step-ahead）
#   --pred_dir    Imputation 预测目录（默认 ../diffusion/output/pred）
#   --no_imputed  跳过 y_imputed 注入（不使用 imputation 先验）
#   --model       选择 forecasting 模型: nsdiff | patchtst | dlinear | itransformer | informer | lstm |
#                                          futuretst | futuretst-windownorm | tide
#                 （默认 nsdiff）

# 默认参数
MODEL=nsdiff
WINDOWS=168
PRED_LEN=18
EPOCHS=200
PATIENCE=20
LR=0.001
NPZ_PATH="../data_processing/data/prepped.npz"
DEVICE="cuda"
SEEDS="[1]"
STEP_INDEX=0
STRIDE=24
BASINS=""
IMPUTE_PRED_DIR="diffusion/output/pred"
NO_IMPUTED=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --windows)    WINDOWS="$2"; shift 2 ;;
        --pred_len)   PRED_LEN="$2"; shift 2 ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --patience)   PATIENCE="$2"; shift 2 ;;
        --lr)         LR="$2"; shift 2 ;;
        --npz_path)   NPZ_PATH="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --step_index) STEP_INDEX="$2"; shift 2 ;;
        --stride)     STRIDE="$2"; shift 2 ;;
        --basins)     BASINS="$2"; shift 2 ;;
        --pred_dir)   IMPUTE_PRED_DIR="$2"; shift 2 ;;
        --no_imputed) NO_IMPUTED=true; shift ;;
        --model)      MODEL="$2"; shift 2 ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIFFUSION_DIR="$SCRIPT_DIR/diffusion_forecast"
DATA_DIR="$SCRIPT_DIR/data_processing"

echo "=========================================="
echo "Forecasting Pipeline"
echo "=========================================="
echo "Model: $MODEL"
echo ""
echo "Configuration:"
echo "  windows   = $WINDOWS"
echo "  pred_len  = $PRED_LEN"
echo "  epochs    = $EPOCHS"
echo "  patience  = $PATIENCE"
echo "  lr        = $LR"
echo "  npz_path  = $NPZ_PATH"
echo "  device    = $DEVICE"
echo "  seeds      = $SEEDS"
echo "  stride     = $STRIDE"
echo "  step_index = $STEP_INDEX"
if [ "$NO_IMPUTED" = false ]; then
echo "  pred_dir   = $IMPUTE_PRED_DIR (imputation predictions)"
else
echo "  y_imputed  = DISABLED (--no_imputed)"
fi
echo "=========================================="
echo ""

# Step 1: Preprocessing
echo "[Step 1/3] Preprocessing (forecast format)..."
echo "=========================================="
cd "$DATA_DIR"
python3 preprocess_camelsh_forecast.py $BASINS
echo ""

# Step 1.5: Inject y_imputed from imputation predictions
if [ "$NO_IMPUTED" = false ]; then
    echo "[Step 1.5/3] Injecting y_imputed from imputation predictions..."
    echo "=========================================="
    python3 fill_forecast_npz.py inject \
        --pred_dir "$SCRIPT_DIR/$IMPUTE_PRED_DIR" \
        --forecast_npz data/prepped.npz
    echo ""
else
    echo "[Step 1.5/3] Skipping y_imputed injection (--no_imputed)"
    echo ""
fi

# 进入 diffusion_forecast 目录
cd "$DIFFUSION_DIR"
export PYTHONPATH=./

# 获取 batch_size（= basin 数量）
BATCH_SIZE=$(python3 -c "\
import numpy as np; \
print(int(np.load('$NPZ_PATH', allow_pickle=True)['n_segs']))")
echo "Batch size (n_segs): $BATCH_SIZE"
echo ""

# 训练
echo "[Step 2/3] Training $MODEL forecasting model..."
echo "=========================================="

# 通用 args：所有 forecast 模型语义一致的流水线参数
COMMON_ARGS=(
    --dataset_type=CAMELS
    --npz_path="$NPZ_PATH"
    --device="$DEVICE"
    --batch_size=$BATCH_SIZE
    --horizon=1
    --windows=$WINDOWS
    --pred_len=$PRED_LEN
    --epochs=$EPOCHS
    --patience=$PATIENCE
    --lr=$LR
)

# 模型 dispatcher：选择实验入口文件 + 模型专属 args（仅那些与 shell 流水线状态强耦合的字段）
case "$MODEL" in
    nsdiff)
        EXP_FILE="src/experiments/NsDiff_CAMELS_forecast.py"
        MODEL_ARGS=(--load_pretrain=False)
        ;;
    patchtst)
        EXP_FILE="src/experiments/PatchTST_forecast.py"
        MODEL_ARGS=()
        ;;
    dlinear)
        EXP_FILE="src/experiments/DLinear_forecast.py"
        MODEL_ARGS=()
        ;;
    itransformer)
        EXP_FILE="src/experiments/iTransformer_forecast.py"
        MODEL_ARGS=()
        ;;
    informer)
        EXP_FILE="src/experiments/Informer_forecast.py"
        MODEL_ARGS=()
        ;;
    lstm)
        EXP_FILE="src/experiments/LSTM_forecast.py"
        MODEL_ARGS=()
        ;;
    futuretst)
        EXP_FILE="src/experiments/FutureTST_forecast.py"
        MODEL_ARGS=()
        ;;
    futuretst-windownorm)
        EXP_FILE="src/experiments/FutureTST_windownorm_forecast.py"
        MODEL_ARGS=()
        ;;
    tide)
        EXP_FILE="src/experiments/TiDE_forecast.py"
        MODEL_ARGS=()
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Supported: nsdiff | patchtst | dlinear | itransformer | informer | lstm | futuretst | futuretst-windownorm | tide"
        exit 1
        ;;
esac

CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 "$EXP_FILE" "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" runs --seeds="$SEEDS"

# Postprocess: 反归一化 + 计算指标
echo ""
echo "[Step 3/3] Postprocessing predictions..."
echo "=========================================="

cd "$SCRIPT_DIR"
python3 data_processing/postprocess_forecast.py \
    --pred_dir "$DIFFUSION_DIR/output/pred" \
    --partition tst \
    --step_index $STEP_INDEX \
    --window $WINDOWS \
    --pred_len $PRED_LEN \
    --stride $STRIDE

echo ""
echo "=========================================="
echo "Forecasting Pipeline Complete!"
echo "=========================================="
echo "Predictions (normalized): $DIFFUSION_DIR/output/pred/"
echo "Predictions (denormalized): $DIFFUSION_DIR/output/denorm/"
echo "Model checkpoints: $DIFFUSION_DIR/results/"
echo "=========================================="
