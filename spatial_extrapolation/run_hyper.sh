#!/bin/bash
# VAE超参数搜索脚本 - 优化版
# 重点探索dropout正则化效果，固定epochs以降低计算成本
# 总共192种超参数组合
#
# 用法:
#   bash run_hyper.sh [NUM_FOLDS] [START_FOLD] [END_FOLD]
#
# 示例:
#   bash run_hyper.sh 106 1 2      # 只运行前2折（快速测试）
#   bash run_hyper.sh 106 1 106    # 全部106折（完整搜索）
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CSV_FILE="$SCRIPT_DIR/../../denormalized_camels_data_time.parquet"

# =============================================================================
# VAE超参数网格定义
# =============================================================================

# 策略：
# - Dropout: 3个值 (0, 0.1, 0.3) - 探索正则化对泛化能力的影响
# - Epochs: 固定为100 - 降低计算成本，专注于架构和正则化的影响
# - 其他超参数: 保留2个代表性值以平衡搜索空间


# 神经网络通用超参数
LATENT_DIMS=(16 32)                      # 中等 vs 较大 (2个值)
HIDDEN_DIMS=(128 256)                    # 标准 vs 大 (2个值)
LEARNING_RATES=(1e-3 5e-3)               # 标准 vs 较快 (2个值)
DROPOUT_RATES=(0 0.1 0.3)                # 无dropout vs 弱 vs 中等 (3个值)
EPOCHS_LIST=(100)                        # 固定训练轮数 (1个值)
BATCH_SIZES=(32 64)                      # 标准 vs 大批次 (2个值)

# VAE特有超参数（每个2个值）
BETA_SCHEDULES=("constant" "linear")     # 两种调度策略 (2个值)
BETA_VALUES=(0.1 0.5)                    # 低 vs 中等 (2个值)

# 总组合数 = 2 × 2 × 2 × 3 × 1 × 2 × 2 × 2 = 192 种

# K-fold设置
NUM_FOLDS=${1:-106}                      # 总fold数
START_FOLD=${2:-1}                       # 起始fold
END_FOLD=${3:-2}                         # 结束fold（默认2，快速测试）

# 结果保存路径
RESULTS_CSV="$SCRIPT_DIR/hyperparam_search.csv"
# =============================================================================
# 初始化结果CSV文件
# =============================================================================

# 如果CSV已存在，备份并创建新的
if [ -f "$RESULTS_CSV" ]; then
    BACKUP_CSV="$RESULTS_CSV.backup_$(date +%Y%m%d_%H%M%S)"
    echo "⚠️  发现已存在的结果文件，备份到: $BACKUP_CSV"
    mv "$RESULTS_CSV" "$BACKUP_CSV"
fi

# 创建新的CSV并写入header（必须与apply_vae_hyperparam.py的列完全一致）
echo "hyperparam_id,latent_dim,hidden_dim,lr,dropout,epochs,beta_schedule,beta_value,batch_size,fold,basin,y_mean_true,y_mean_vae,y_std_true,y_std_vae,mean_error_pct,std_error_pct,timestamp" > "$RESULTS_CSV"
echo "✅ 初始化结果CSV: $RESULTS_CSV"
echo ""

# =============================================================================
# Basin分割
# =============================================================================

ALL_BASINS=$(python3 -c "\
import pandas as pd; \
df = pd.read_parquet('$CSV_FILE', columns=['basin_id']); \
print(' '.join(sorted(df['basin_id'].unique())))")
BASIN_ARRAY=($ALL_BASINS)
TOTAL_BASINS=${#BASIN_ARRAY[@]}

# 验证fold数量
if [ $NUM_FOLDS -gt $TOTAL_BASINS ]; then
    echo "ERROR: NUM_FOLDS ($NUM_FOLDS) 不能大于 TOTAL_BASINS ($TOTAL_BASINS)"
    exit 1
fi

# 计算每份的大小
BASINS_PER_FOLD=$((TOTAL_BASINS / NUM_FOLDS))
REMAINDER=$((TOTAL_BASINS % NUM_FOLDS))

# =============================================================================
# 打印搜索配置
# =============================================================================

TOTAL_COMBINATIONS=$((${#LATENT_DIMS[@]} * ${#HIDDEN_DIMS[@]} * ${#LEARNING_RATES[@]} * ${#DROPOUT_RATES[@]} * ${#EPOCHS_LIST[@]} * ${#BATCH_SIZES[@]} * ${#BETA_SCHEDULES[@]} * ${#BETA_VALUES[@]}))

echo "=========================================="
echo "VAE超参数搜索 - 优化版"
echo "=========================================="
echo ""
echo "搜索策略: 探索dropout正则化效果，固定epochs降低计算成本"
echo ""
echo "超参数范围:"
echo "  latent_dims: ${LATENT_DIMS[@]}"
echo "  hidden_dims: ${HIDDEN_DIMS[@]}"
echo "  learning_rates: ${LEARNING_RATES[@]}"
echo "  dropout_rates: ${DROPOUT_RATES[@]}"
echo "  epochs: ${EPOCHS_LIST[@]}"
echo "  batch_sizes: ${BATCH_SIZES[@]}"
echo "  beta_schedules: ${BETA_SCHEDULES[@]}"
echo "  beta_values: ${BETA_VALUES[@]}"
echo ""
echo "总超参数组合数: $TOTAL_COMBINATIONS"
echo ""
echo "K-fold设置:"
echo "  总basin数: $TOTAL_BASINS"
echo "  总fold数: $NUM_FOLDS"
echo "  运行fold范围: $START_FOLD - $END_FOLD"
echo "  每fold的basin数: ~$BASINS_PER_FOLD"
if [ $REMAINDER -gt 0 ]; then
    echo "  注意: 前 $REMAINDER 个fold会有 $((BASINS_PER_FOLD + 1)) 个basin"
fi
echo ""
echo "结果保存在: $RESULTS_CSV"
echo ""

# =============================================================================
# 超参数搜索主循环
# =============================================================================

HYPERPARAM_ID=0

# 8层嵌套循环
for latent_dim in "${LATENT_DIMS[@]}"; do
  for hidden_dim in "${HIDDEN_DIMS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
      for dropout in "${DROPOUT_RATES[@]}"; do
        for epochs in "${EPOCHS_LIST[@]}"; do
          for batch_size in "${BATCH_SIZES[@]}"; do
            for beta_schedule in "${BETA_SCHEDULES[@]}"; do
              for beta_value in "${BETA_VALUES[@]}"; do

                HYPERPARAM_ID=$((HYPERPARAM_ID + 1))

                echo "=========================================="
                echo "超参数组合 $HYPERPARAM_ID/$TOTAL_COMBINATIONS"
                echo "=========================================="
                echo "  latent_dim: $latent_dim"
                echo "  hidden_dim: $hidden_dim"
                echo "  lr: $lr"
                echo "  dropout: $dropout"
                echo "  epochs: $epochs"
                echo "  batch_size: $batch_size"
                echo "  beta_schedule: $beta_schedule"
                echo "  beta_value: $beta_value"
                echo ""


                # K-fold交叉验证
                START_IDX=0

                # 计算起始索引
                for fold in $(seq 1 $((START_FOLD - 1))); do
                  if [ $fold -le $REMAINDER ]; then
                    START_IDX=$((START_IDX + BASINS_PER_FOLD + 1))
                  else
                    START_IDX=$((START_IDX + BASINS_PER_FOLD))
                  fi
                done

                # 运行指定范围的fold
                for fold in $(seq $START_FOLD $END_FOLD); do
                  echo "------------------------------------------"
                  echo "  运行 Fold $fold/$NUM_FOLDS"
                  echo "------------------------------------------"

                  # 计算当前fold的大小
                  if [ $fold -le $REMAINDER ]; then
                    CURRENT_FOLD_SIZE=$((BASINS_PER_FOLD + 1))
                  else
                    CURRENT_FOLD_SIZE=$BASINS_PER_FOLD
                  fi

                  END_IDX=$((START_IDX + CURRENT_FOLD_SIZE))

                  # 提取当前fold的basin ID
                  FOLD_BASINS="${BASIN_ARRAY[@]:$START_IDX:$CURRENT_FOLD_SIZE}"

                  echo "  处理 $CURRENT_FOLD_SIZE 个target basin (索引 $START_IDX 到 $((END_IDX-1)))"
                  if [ $CURRENT_FOLD_SIZE -le 3 ]; then
                    echo "  Target basin(s): $FOLD_BASINS"
                  else
                    echo "  前3个basins: $(echo $FOLD_BASINS | cut -d' ' -f1-3)"
                  fi
                  echo ""

                  # 调用训练脚本
                  bash "$SCRIPT_DIR/run_camels_perstd_hyperparam.sh" \
                    --latent_dim $latent_dim \
                    --hidden_dim $hidden_dim \
                    --lr $lr \
                    --dropout $dropout \
                    --epochs $epochs \
                    --batch_size $batch_size \
                    --beta_schedule $beta_schedule \
                    --beta_value $beta_value \
                    --hyperparam_id $HYPERPARAM_ID \
                    --fold $fold \
                    --results_csv "$RESULTS_CSV" \
                    $FOLD_BASINS

                  echo ""
                  echo "  Fold $fold/$NUM_FOLDS 完成!"

                  START_IDX=$END_IDX
                done

                # 汇总当前超参数组合的结果
                echo ""
                echo "=========================================="
                echo "超参数组合 $HYPERPARAM_ID 完成"
                echo "=========================================="

                # 计算该超参数组合的平均性能
                python3 - <<EOF
import pandas as pd
import sys

try:
    df = pd.read_csv('$RESULTS_CSV')
    df_current = df[df['hyperparam_id'] == $HYPERPARAM_ID]

    if len(df_current) > 0:
        mean_error = df_current['mean_error_pct'].astype(float).mean()
        std_error = df_current['std_error_pct'].astype(float).mean()
        print(f"  平均mean误差: {mean_error:.2f}%")
        print(f"  平均std误差: {std_error:.2f}%")
        print(f"  测试basin数: {len(df_current)}")
    else:
        print("  未找到结果数据")
except Exception as e:
    print(f"  汇总时出错: {e}")
EOF

                echo ""

              done
            done
          done
        done
      done
    done
  done
done


echo "运行以下命令查看最佳超参数:"
echo "  python $SCRIPT_DIR/summarize_hyperparam_balanced.py"
echo ""
