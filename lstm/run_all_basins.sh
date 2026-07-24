#!/bin/bash

# Basin list
BASINS=(
    "BSR" "CAU" "CRY" "DCR" "DIL" 
    "ECH" "ECR" "FGR" "FON" "GMR" 
    "HYR" "JOR" "JVR" "LCR" "LEM" 
    "MCP" "MCR" "NAV" "PIN" "RFR" 
    "RID" "ROC" "RUE" "SCO" "SJR" 
    "STE" "TPR" "USR" "VAL"
)

# Loop through each basin
for BASIN in "${BASINS[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting basin: $BASIN"
    echo "Time: $(date)"
    echo "========================================"
    
    # Run run.sh
    bash run.sh "$BASIN"
    
    echo "Finished basin: $BASIN at $(date)"
done

echo ""
echo "========================================"
echo "All basins completed!"
echo "Check output/basin_metrics_log_trn.csv and output/basin_metrics_log_tst.csv for results"
echo "========================================"