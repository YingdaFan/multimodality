#!/bin/bash
# K-fold cross-validation for STID imputation
# Usage:
#   bash run_camels_kfold.sh       # Default: 531-fold (leave-one-out, best quality)
#   bash run_camels_kfold.sh 53    # 53-fold (good quality, ~29 hours)
#   bash run_camels_kfold.sh 10    # 10-fold (fast validation, ~5-6 hours)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CSV_FILE="../../denormalized_camels_data_time.parquet"

# 默认为531-fold（leave-one-out）
NUM_FOLDS=${1:-106}

echo "=========================================="
echo "Starting ${NUM_FOLDS}-Fold Cross-Validation"
echo "=========================================="

# 从CSV提取所有unique basin ID（保持前导零）
echo "Extracting unique basin IDs..."
ALL_BASINS=$(python3 -c "\
import pandas as pd; \
df = pd.read_parquet('$CSV_FILE', columns=['basin_id']); \
print(' '.join(sorted(df['basin_id'].unique())))")
BASIN_ARRAY=($ALL_BASINS)
TOTAL_BASINS=${#BASIN_ARRAY[@]}

echo "Total basins found: $TOTAL_BASINS"
echo "Number of folds: $NUM_FOLDS"
echo ""

# 验证fold数量
if [ $NUM_FOLDS -gt $TOTAL_BASINS ]; then
    echo "ERROR: NUM_FOLDS ($NUM_FOLDS) cannot be greater than TOTAL_BASINS ($TOTAL_BASINS)"
    exit 1
fi

# 计算每份的大小
BASINS_PER_FOLD=$((TOTAL_BASINS / NUM_FOLDS))
REMAINDER=$((TOTAL_BASINS % NUM_FOLDS))

echo "Basins per fold: ~$BASINS_PER_FOLD"
if [ $REMAINDER -gt 0 ]; then
    echo "Note: First $REMAINDER folds will have $((BASINS_PER_FOLD + 1)) basins"
fi


echo ""


# 循环K次
START_IDX=0
for fold in $(seq 1 $NUM_FOLDS); do
    echo "=========================================="
    echo "Running Fold $fold/$NUM_FOLDS"
    echo "=========================================="

    # 计算当前fold的大小（前几个fold多分配remainder）
    if [ $fold -le $REMAINDER ]; then
        CURRENT_FOLD_SIZE=$((BASINS_PER_FOLD + 1))
    else
        CURRENT_FOLD_SIZE=$BASINS_PER_FOLD
    fi

    END_IDX=$((START_IDX + CURRENT_FOLD_SIZE))

    # 提取当前fold的basin ID
    FOLD_BASINS="${BASIN_ARRAY[@]:$START_IDX:$CURRENT_FOLD_SIZE}"

    echo "Processing $CURRENT_FOLD_SIZE target basin(s) (indices $START_IDX to $((END_IDX-1)))"
    if [ $CURRENT_FOLD_SIZE -le 5 ]; then
        echo "Target basin(s): $FOLD_BASINS"
    else
        echo "First 5 basins: $(echo $FOLD_BASINS | cut -d' ' -f1-5)"
    fi
    echo ""

    # 调用run_camels.sh，传入当前fold的basin ID列表
    bash "$SCRIPT_DIR/run_camels_perstd.sh" $FOLD_BASINS

    echo ""
    echo "Fold $fold/$NUM_FOLDS completed!"


    START_IDX=$END_IDX
done

echo "=========================================="
echo "${NUM_FOLDS}-Fold Cross-Validation Complete!"
echo "=========================================="
echo "All results saved in basin_metrics_log.csv"

