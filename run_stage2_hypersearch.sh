#!/bin/bash
# Stage 2 Hyperparameter Sensitivity Analysis (ICML Style)
#
# Design inspired by TimeDart (ICML 2025):
# - Group experiments by module/component
# - Control variable method: change one parameter at a time
# - Include baseline (default config) for comparison
# - Focus on trends and robustness, not exhaustive grid search
#
# This script:
# 1. Runs Stage 1 ONCE to prepare prepped.npz
# 2. Runs grouped hyperparameter experiments
#
# Usage:
#   bash run_stage2_hypersearch.sh <basin_id>
#   bash run_stage2_hypersearch.sh <basin_id> <group_name>  # Run specific group only
#   bash run_stage2_hypersearch.sh --skip-stage1 <basin_id> <group_name>  # Skip Stage 1
#
# Groups:
#   all        - Run all groups (default)
#   baseline   - Baseline only
#   diffusion  - Diffusion steps
#   capacity   - Model capacity (d_model, e_layers)
#   learning   - Learning rate
#   fusion     - Fusion type (ablation)

if [ $# -eq 0 ]; then
    echo "Usage: bash run_stage2_hypersearch.sh [--skip-stage1] <basin_id> [group_name]"
    echo ""
    echo "Options:"
    echo "  --skip-stage1  Skip Stage 1 (use existing prepped.npz)"
    echo ""
    echo "Groups: all, baseline, diffusion, capacity, learning, fusion"
    echo ""
    echo "Examples:"
    echo "  bash run_stage2_hypersearch.sh 01022500              # Run Stage 1 + all groups"
    echo "  bash run_stage2_hypersearch.sh 01022500 diffusion    # Run Stage 1 + diffusion group"
    echo "  bash run_stage2_hypersearch.sh --skip-stage1 01022500 capacity  # Skip Stage 1, run capacity"
    exit 1
fi

# Parse --skip-stage1 option
SKIP_STAGE1=false
if [ "$1" == "--skip-stage1" ]; then
    SKIP_STAGE1=true
    shift
fi

BASIN_ID=$1
GROUP=${2:-all}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LSTM_DIR="$SCRIPT_DIR/lstm"
DIFFUSION_DIR="$SCRIPT_DIR/diffusion"
DATA_DIR="$SCRIPT_DIR/data_processing"

echo "=========================================="
echo "Stage 2 Hyperparameter Sensitivity Analysis"
echo "=========================================="
echo "Target Basin: $BASIN_ID"
echo "Experiment Group: $GROUP"
echo ""

# ==========================================
# Stage 1: Run ONCE to prepare prepped.npz
# ==========================================
if [ "$SKIP_STAGE1" = true ]; then
    echo "=========================================="
    echo "Stage 1: SKIPPED (--skip-stage1 flag set)"
    echo "=========================================="
    echo "Using existing prepped.npz"

    # Verify prepped.npz exists
    PREPPED_NPZ="$DATA_DIR/data/prepped.npz"
    if [ ! -f "$PREPPED_NPZ" ]; then
        echo "ERROR: prepped.npz not found at $PREPPED_NPZ"
        echo "Please run without --skip-stage1 first."
        exit 1
    fi
    echo "Found: $PREPPED_NPZ"
else
    echo "=========================================="
    echo "Stage 1: Preparing prepped.npz"
    echo "=========================================="
    bash "$LSTM_DIR/run_camels_perstd_stage1_raw.sh" $BASIN_ID

    echo ""
    echo "Stage 1 complete. prepped.npz is ready."
fi
echo ""

# ==========================================
# Stage 2: Hyperparameter Experiments
# ==========================================
cd "$DIFFUSION_DIR"
export PYTHONPATH=./

# Output directory
HYPERSEARCH_OUTPUT="$DIFFUSION_DIR/output/hypersearch/${BASIN_ID}"
mkdir -p "$HYPERSEARCH_OUTPUT"

# Log file with metrics
LOG_FILE="$HYPERSEARCH_OUTPUT/experiment_results.csv"

# Initialize log file only if:
# 1. Running all groups AND not skipping stage1 (fresh start), OR
# 2. Log file doesn't exist
if [[ ! -f "$LOG_FILE" ]] || [[ "$GROUP" == "all" && "$SKIP_STAGE1" = false ]]; then
    echo "group,exp_name,d_model,n_heads,e_layers,d_ff,diffusion_steps,lr,dropout,fusion_type,nse,kge,rmse,mae,r2,pbias,status" > "$LOG_FILE"
    echo "Initialized new log file: $LOG_FILE"
else
    echo "Appending to existing log file: $LOG_FILE"
fi

# ==========================================
# Default (Baseline) Configuration
# ==========================================
DEFAULT_D_MODEL=512
DEFAULT_N_HEADS=8
DEFAULT_E_LAYERS=2
DEFAULT_D_FF=1024
DEFAULT_DIFFUSION_STEPS=20
DEFAULT_LR=0.001
DEFAULT_DROPOUT=0.05
DEFAULT_FUSION_TYPE=""

# ==========================================
# Helper Function: Run Single Experiment
# ==========================================
run_experiment() {
    local GROUP_NAME=$1
    local EXP_NAME=$2
    local D_MODEL=$3
    local N_HEADS=$4
    local E_LAYERS=$5
    local D_FF=$6
    local DIFF_STEPS=$7
    local LR=$8
    local DROPOUT=$9
    local FUSION_TYPE=${10}

    echo ""
    echo "----------------------------------------"
    echo "[$GROUP_NAME] $EXP_NAME"
    echo "----------------------------------------"
    echo "  d_model=$D_MODEL, n_heads=$N_HEADS, e_layers=$E_LAYERS"
    echo "  d_ff=$D_FF, diffusion_steps=$DIFF_STEPS"
    echo "  lr=$LR, dropout=$DROPOUT, fusion=$FUSION_TYPE"

    # Create experiment output directory
    EXP_OUTPUT="$HYPERSEARCH_OUTPUT/${GROUP_NAME}/${EXP_NAME}"
    mkdir -p "$EXP_OUTPUT"

    # Get batch size from prepped.npz (= number of basins)
    NPZ_PATH="../data_processing/data/prepped.npz"
    BATCH_SIZE=$(python3 -c "\
import numpy as np; \
print(int(np.load('$NPZ_PATH', allow_pickle=True)['n_segs']))")

    # Run experiment
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    python3 ./src/experiments/diffcal_gx_enc.py \
        --dataset_type="CAMELS" \
        --npz_path="$NPZ_PATH" \
        --device="cuda" \
        --batch_size=$BATCH_SIZE \
        --horizon=1 \
        --pred_len=365 \
        --windows=365 \
        --load_pretrain=False \
        --epochs=200 \
        --patience=20 \
        --lr=$LR \
        --d_model=$D_MODEL \
        --n_heads=$N_HEADS \
        --e_layers=$E_LAYERS \
        --d_ff=$D_FF \
        --diffusion_steps=$DIFF_STEPS \
        --dropout=$DROPOUT \
        --fusion_type="$FUSION_TYPE" \
        --masked_basin_ids $BASIN_ID \
        runs --seeds='[1]' \
        2>&1 | tee "$EXP_OUTPUT/train.log"

    # Check training status
    TRAIN_STATUS=$?

    # Initialize metrics
    NSE="NA"
    KGE="NA"
    RMSE="NA"
    MAE="NA"
    R2="NA"
    PBIAS="NA"
    STATUS="failed"

    if [ $TRAIN_STATUS -eq 0 ] && [ -f "$DIFFUSION_DIR/output/pred/trn.npy" ]; then
        echo ""
        echo "Running evaluation..."

        # Run evaluation script and capture output
        EVAL_OUTPUT=$(cd "$DATA_DIR" && python postprocess_perseg_aligntime_raw.py \
            --pred_dir "$DIFFUSION_DIR/output/pred" \
            --model_name "hypersearch" \
            --partition trn \
            --target_basin "$BASIN_ID" 2>&1)

        echo "$EVAL_OUTPUT" > "$EXP_OUTPUT/eval.log"

        # Extract metrics from output using grep
        # Format: "  NSE: 0.123456"
        NSE=$(echo "$EVAL_OUTPUT" | grep -oP "NSE: \K[-0-9.]+")
        KGE=$(echo "$EVAL_OUTPUT" | grep -oP "KGE: \K[-0-9.]+")
        RMSE=$(echo "$EVAL_OUTPUT" | grep -oP "RMSE: \K[-0-9.]+")
        MAE=$(echo "$EVAL_OUTPUT" | grep -oP "MAE: \K[-0-9.]+")
        R2=$(echo "$EVAL_OUTPUT" | grep -oP "R²: \K[-0-9.]+")
        PBIAS=$(echo "$EVAL_OUTPUT" | grep -oP "PBIAS: \K[-0-9.]+")

        # Use first match if multiple (target basin metrics)
        NSE=$(echo "$NSE" | head -1)
        KGE=$(echo "$KGE" | head -1)
        RMSE=$(echo "$RMSE" | head -1)
        MAE=$(echo "$MAE" | head -1)
        R2=$(echo "$R2" | head -1)
        PBIAS=$(echo "$PBIAS" | head -1)

        # Set defaults if extraction failed
        [ -z "$NSE" ] && NSE="NA"
        [ -z "$KGE" ] && KGE="NA"
        [ -z "$RMSE" ] && RMSE="NA"
        [ -z "$MAE" ] && MAE="NA"
        [ -z "$R2" ] && R2="NA"
        [ -z "$PBIAS" ] && PBIAS="NA"

        STATUS="success"

        echo ""
        echo "Target basin ($BASIN_ID) metrics:"
        echo "  NSE=$NSE, KGE=$KGE, RMSE=$RMSE"
        echo "  MAE=$MAE, R²=$R2, PBIAS=$PBIAS"
    fi

    # Log result with metrics
    echo "$GROUP_NAME,$EXP_NAME,$D_MODEL,$N_HEADS,$E_LAYERS,$D_FF,$DIFF_STEPS,$LR,$DROPOUT,$FUSION_TYPE,$NSE,$KGE,$RMSE,$MAE,$R2,$PBIAS,$STATUS" >> "$LOG_FILE"

    echo "[$GROUP_NAME] $EXP_NAME: $STATUS (NSE=$NSE)"
}

# ==========================================
# Group 0: Baseline
# ==========================================
run_baseline() {
    echo ""
    echo "=========================================="
    echo "Group 0: Baseline (Default Configuration)"
    echo "=========================================="

    run_experiment "baseline" "default" \
        $DEFAULT_D_MODEL $DEFAULT_N_HEADS $DEFAULT_E_LAYERS $DEFAULT_D_FF \
        $DEFAULT_DIFFUSION_STEPS $DEFAULT_LR $DEFAULT_DROPOUT "$DEFAULT_FUSION_TYPE"
}

# ==========================================
# Group 1: Diffusion Steps
# Analysis: Impact of denoising steps on generation quality
# ==========================================
run_diffusion_group() {
    echo ""
    echo "=========================================="
    echo "Group 1: Diffusion Steps Analysis"
    echo "=========================================="
    echo "Testing: diffusion_steps in {10, 20, 50}"
    echo "Fixed: d_model=$DEFAULT_D_MODEL, e_layers=$DEFAULT_E_LAYERS, lr=$DEFAULT_LR"

    for DIFF_STEPS in 10 20 50; do
        run_experiment "diffusion" "steps_${DIFF_STEPS}" \
            $DEFAULT_D_MODEL $DEFAULT_N_HEADS $DEFAULT_E_LAYERS $DEFAULT_D_FF \
            $DIFF_STEPS $DEFAULT_LR $DEFAULT_DROPOUT "$DEFAULT_FUSION_TYPE"
    done
}

# ==========================================
# Group 2: Model Capacity
# Analysis: Impact of model size on performance
# ==========================================
run_capacity_group() {
    echo ""
    echo "=========================================="
    echo "Group 2: Model Capacity Analysis"
    echo "=========================================="
    echo "Testing: (d_model, e_layers) combinations"
    echo "Fixed: diffusion_steps=$DEFAULT_DIFFUSION_STEPS, lr=$DEFAULT_LR"

    # d_model variations (with proportional d_ff = 2*d_model)
    for D_MODEL in 256 512; do
        D_FF=$((D_MODEL * 2))
        N_HEADS=$((D_MODEL / 64))  # Scale heads with d_model
        [ $N_HEADS -lt 4 ] && N_HEADS=4

        for E_LAYERS in 2 3 4; do
            run_experiment "capacity" "dm${D_MODEL}_el${E_LAYERS}" \
                $D_MODEL $N_HEADS $E_LAYERS $D_FF \
                $DEFAULT_DIFFUSION_STEPS $DEFAULT_LR $DEFAULT_DROPOUT "$DEFAULT_FUSION_TYPE"
        done
    done
}

# ==========================================
# Group 3: Learning Rate
# Analysis: Convergence and optimization
# ==========================================
run_learning_group() {
    echo ""
    echo "=========================================="
    echo "Group 3: Learning Rate Analysis"
    echo "=========================================="
    echo "Testing: lr in {0.001, 0.0005, 0.0001}"
    echo "Fixed: d_model=$DEFAULT_D_MODEL, e_layers=$DEFAULT_E_LAYERS"

    for LR in 0.001 0.0005 0.0001; do
        run_experiment "learning" "lr_${LR}" \
            $DEFAULT_D_MODEL $DEFAULT_N_HEADS $DEFAULT_E_LAYERS $DEFAULT_D_FF \
            $DEFAULT_DIFFUSION_STEPS $LR $DEFAULT_DROPOUT "$DEFAULT_FUSION_TYPE"
    done
}

# ==========================================
# Group 4: Fusion Type (Ablation Study)
# Analysis: Effect of wave fusion strategies
# ==========================================
run_fusion_group() {
    echo ""
    echo "=========================================="
    echo "Group 4: Fusion Type Ablation"
    echo "=========================================="
    echo "Testing: fusion_type in {none, scatter, interference}"
    echo "Fixed: all other parameters at default"

    # No fusion (baseline)
    run_experiment "fusion" "none" \
        $DEFAULT_D_MODEL $DEFAULT_N_HEADS $DEFAULT_E_LAYERS $DEFAULT_D_FF \
        $DEFAULT_DIFFUSION_STEPS $DEFAULT_LR $DEFAULT_DROPOUT ""

    # Scatter fusion
    run_experiment "fusion" "scatter" \
        $DEFAULT_D_MODEL $DEFAULT_N_HEADS $DEFAULT_E_LAYERS $DEFAULT_D_FF \
        $DEFAULT_DIFFUSION_STEPS $DEFAULT_LR $DEFAULT_DROPOUT "scatter"

    # Interference fusion
    run_experiment "fusion" "interference" \
        $DEFAULT_D_MODEL $DEFAULT_N_HEADS $DEFAULT_E_LAYERS $DEFAULT_D_FF \
        $DEFAULT_DIFFUSION_STEPS $DEFAULT_LR $DEFAULT_DROPOUT "interference"
}

# ==========================================
# Run Selected Groups
# ==========================================
case $GROUP in
    all)
        run_baseline
        run_diffusion_group
        run_capacity_group
        run_learning_group
        run_fusion_group
        ;;
    baseline)
        run_baseline
        ;;
    diffusion)
        run_diffusion_group
        ;;
    capacity)
        run_capacity_group
        ;;
    learning)
        run_learning_group
        ;;
    fusion)
        run_fusion_group
        ;;
    *)
        echo "Unknown group: $GROUP"
        echo "Available groups: all, baseline, diffusion, capacity, learning, fusion"
        exit 1
        ;;
esac

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Hyperparameter Analysis Complete!"
echo "=========================================="
echo "Basin: $BASIN_ID"
echo "Group: $GROUP"
echo ""
echo "Experiment Summary:"
echo "  - baseline:  1 experiment  (default configuration)"
echo "  - diffusion: 3 experiments (steps: 10, 20, 50)"
echo "  - capacity:  6 experiments (d_model x e_layers)"
echo "  - learning:  3 experiments (lr: 0.001, 0.0005, 0.0001)"
echo "  - fusion:    3 experiments (none, scatter, interference)"
echo "  - Total:    16 experiments"
echo ""
echo "Results saved in: $HYPERSEARCH_OUTPUT"
echo ""
echo "Output files:"
echo "  - $LOG_FILE"
echo "    (hyperparameters + metrics: NSE, KGE, RMSE, MAE, R², PBIAS)"
echo ""
echo "To analyze results:"
echo "  cat $LOG_FILE | column -t -s ','"
echo ""
echo "To sort by NSE (best first):"
echo "  sort -t',' -k11 -rn $LOG_FILE | head"
echo "=========================================="
