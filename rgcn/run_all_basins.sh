#!/bin/bash

# Basin列表
BASINS=(
    "BSR" "CAU" "CRY" "DCR" "DIL" 
    "ECH" "ECR" "FGR" "FON" "GMR" 
    "HYR" "JOR" "JVR" "LCR" "LEM" 
    "MCP" "MCR" "NAV" "PIN" "RFR" 
    "RID" "ROC" "RUE" "SCO" "SJR" 
    "STE" "TPR" "USR" "VAL"
)

# 循环运行每个basin
for BASIN in "${BASINS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting basin: $BASIN"
    echo "Time: $(date)"
    echo "========================================"
    
    # 运行run.sh，它会等待base.py和evaluate.py都完成
    bash run.sh "$BASIN"
    
    echo "Finished basin: $BASIN at $(date)"
done

echo ""
echo "========================================"
echo "All basins completed!"
echo "Check basin_metrics_log.csv for results"
echo "========================================"