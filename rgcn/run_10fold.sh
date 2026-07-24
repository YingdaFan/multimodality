#!/bin/bash
# 10-fold cross-validation - 简化版
# 直接从CSV读取basin ID，分成10份，依次调用 run_camels_allbasinstd.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CSV_FILE="$SCRIPT_DIR/../../denormalized_camels_data_time.parquet"

echo "=========================================="
echo "Starting 10-Fold Cross-Validation"
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
echo ""

# 计算每份的大小
BASINS_PER_FOLD=$((TOTAL_BASINS / 10))
REMAINDER=$((TOTAL_BASINS % 10))

echo "Basins per fold: ~$BASINS_PER_FOLD"
echo ""

# 循环10次
START_IDX=0
for fold in {1..10}; do
    echo "=========================================="
    echo "Running Fold $fold/10"
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

    echo "Masking $CURRENT_FOLD_SIZE basins (indices $START_IDX to $((END_IDX-1)))"
    echo "First 5 basins: $(echo $FOLD_BASINS | cut -d' ' -f1-5)"
    echo ""

    # 调用原始脚本，传入basin ID列表
    bash "$SCRIPT_DIR/run_camels_allbasinstd.sh" $FOLD_BASINS

    echo ""
    echo "Fold $fold completed!"
    echo ""

    START_IDX=$END_IDX
done

echo "=========================================="
echo "10-Fold Cross-Validation Complete!"
echo "=========================================="
echo "All results saved in basin_metrics_log.csv"
