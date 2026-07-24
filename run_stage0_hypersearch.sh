#!/bin/bash
# Stage 0 (VAE) Hyperparameter Sensitivity Analysis
#
# VAE is used to predict y_mean and y_std for masked basins.
# This script searches for optimal VAE hyperparameters.
#
# Usage:
#   bash run_stage0_hypersearch.sh <basin_id>
#   bash run_stage0_hypersearch.sh <basin_id> <group_name>
#   bash run_stage0_hypersearch.sh --skip-preprocess <basin_id> <group_name>
#
# Groups:
#   all        - Run all groups (default)
#   baseline   - Baseline only
#   capacity   - Model capacity (latent_dim, hidden_dim)
#   learning   - Learning rate
#   regularization - Dropout and beta

if [ $# -eq 0 ]; then
    echo "Usage: bash run_stage0_hypersearch.sh [--skip-preprocess] <basin_id> [group_name]"
    echo ""
    echo "Groups: all, baseline, capacity, learning, regularization"
    exit 1
fi

# Parse --skip-preprocess option
SKIP_PREPROCESS=false
if [ "$1" == "--skip-preprocess" ]; then
    SKIP_PREPROCESS=true
    shift
fi

BASIN_ID=$1
GROUP=${2:-all}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LSTM_DIR="$SCRIPT_DIR/lstm"
DATA_DIR="$SCRIPT_DIR/data_processing"

echo "=========================================="
echo "Stage 0 (VAE) Hyperparameter Analysis"
echo "=========================================="
echo "Target Basin: $BASIN_ID"
echo "Experiment Group: $GROUP"
echo ""

# ==========================================
# Preprocessing: Run ONCE
# ==========================================
if [ "$SKIP_PREPROCESS" = true ]; then
    echo "Preprocessing: SKIPPED"
else
    echo "Preprocessing data..."
    cd "$DATA_DIR"
    python preprocess_perseg_aligntime_camels.py
    python modify_basin_to_nan_allmask.py $BASIN_ID
fi
echo ""

# ==========================================
# Output Directory
# ==========================================
HYPERSEARCH_OUTPUT="$DATA_DIR/output/hypersearch_vae/${BASIN_ID}"
mkdir -p "$HYPERSEARCH_OUTPUT"

# Log file
LOG_FILE="$HYPERSEARCH_OUTPUT/experiment_results.csv"

if [[ ! -f "$LOG_FILE" ]] || [[ "$GROUP" == "all" && "$SKIP_PREPROCESS" = false ]]; then
    echo "group,exp_name,latent_dim,hidden_dim,dropout,epochs,lr,beta_value,mean_err_pct,std_err_pct,status" > "$LOG_FILE"
fi

# ==========================================
# Default Configuration
# ==========================================
DEFAULT_LATENT_DIM=16
DEFAULT_HIDDEN_DIM=128
DEFAULT_DROPOUT=0.0
DEFAULT_EPOCHS=100
DEFAULT_LR=0.001
DEFAULT_BETA=0.1

# ==========================================
# Helper: Run Single Experiment
# ==========================================
run_experiment() {
    local GROUP_NAME=$1
    local EXP_NAME=$2
    local LATENT_DIM=$3
    local HIDDEN_DIM=$4
    local DROPOUT=$5
    local EPOCHS=$6
    local LR=$7
    local BETA=$8

    echo ""
    echo "----------------------------------------"
    echo "[$GROUP_NAME] $EXP_NAME"
    echo "----------------------------------------"

    EXP_OUTPUT="$HYPERSEARCH_OUTPUT/${GROUP_NAME}/${EXP_NAME}"
    mkdir -p "$EXP_OUTPUT"

    cd "$DATA_DIR"
    EVAL_OUTPUT=$(python apply_vae.py $BASIN_ID \
        --script_dir "$LSTM_DIR" \
        --latent_dim $LATENT_DIM \
        --hidden_dim $HIDDEN_DIM \
        --dropout $DROPOUT \
        --epochs $EPOCHS \
        --lr $LR \
        --beta_value $BETA 2>&1)

    echo "$EVAL_OUTPUT" > "$EXP_OUTPUT/vae.log"

    # Extract metrics (mean_err_pct and std_err_pct for target basin)
    # Use word boundary to avoid partial matches (e.g., "0" matching "10")
    MEAN_ERR=$(echo "$EVAL_OUTPUT" | grep -w "^${BASIN_ID}" | tail -1 | awk '{print $(NF-1)}')
    STD_ERR=$(echo "$EVAL_OUTPUT" | grep -w "^${BASIN_ID}" | tail -1 | awk '{print $NF}')

    [ -z "$MEAN_ERR" ] && MEAN_ERR="NA"
    [ -z "$STD_ERR" ] && STD_ERR="NA"

    if [ "$MEAN_ERR" != "NA" ]; then
        STATUS="success"
        echo "  Mean Error: ${MEAN_ERR}%, Std Error: ${STD_ERR}%"
    else
        STATUS="failed"
    fi

    echo "$GROUP_NAME,$EXP_NAME,$LATENT_DIM,$HIDDEN_DIM,$DROPOUT,$EPOCHS,$LR,$BETA,$MEAN_ERR,$STD_ERR,$STATUS" >> "$LOG_FILE"
    echo "[$GROUP_NAME] $EXP_NAME: $STATUS"
}

# ==========================================
# Group 0: Baseline
# ==========================================
run_baseline() {
    echo ""
    echo "=========================================="
    echo "Group 0: Baseline"
    echo "=========================================="
    run_experiment "baseline" "default" \
        $DEFAULT_LATENT_DIM $DEFAULT_HIDDEN_DIM $DEFAULT_DROPOUT \
        $DEFAULT_EPOCHS $DEFAULT_LR $DEFAULT_BETA
}

# ==========================================
# Group 1: Model Capacity
# ==========================================
run_capacity_group() {
    echo ""
    echo "=========================================="
    echo "Group 1: Model Capacity"
    echo "=========================================="

    for LATENT_DIM in 8 16 32; do
        for HIDDEN_DIM in 64 128 256; do
            run_experiment "capacity" "lat${LATENT_DIM}_hid${HIDDEN_DIM}" \
                $LATENT_DIM $HIDDEN_DIM $DEFAULT_DROPOUT \
                $DEFAULT_EPOCHS $DEFAULT_LR $DEFAULT_BETA
        done
    done
}

# ==========================================
# Group 2: Learning Rate
# ==========================================
run_learning_group() {
    echo ""
    echo "=========================================="
    echo "Group 2: Learning Rate"
    echo "=========================================="

    for LR in 0.01 0.005 0.001 0.0005; do
        run_experiment "learning" "lr_${LR}" \
            $DEFAULT_LATENT_DIM $DEFAULT_HIDDEN_DIM $DEFAULT_DROPOUT \
            $DEFAULT_EPOCHS $LR $DEFAULT_BETA
    done
}

# ==========================================
# Group 3: Regularization (Dropout + Beta)
# ==========================================
run_regularization_group() {
    echo ""
    echo "=========================================="
    echo "Group 3: Regularization"
    echo "=========================================="

    # Dropout
    for DROPOUT in 0.0 0.1 0.2; do
        run_experiment "regularization" "dropout_${DROPOUT}" \
            $DEFAULT_LATENT_DIM $DEFAULT_HIDDEN_DIM $DROPOUT \
            $DEFAULT_EPOCHS $DEFAULT_LR $DEFAULT_BETA
    done

    # Beta (KL weight)
    for BETA in 0.01 0.1 0.5 1.0; do
        run_experiment "regularization" "beta_${BETA}" \
            $DEFAULT_LATENT_DIM $DEFAULT_HIDDEN_DIM $DEFAULT_DROPOUT \
            $DEFAULT_EPOCHS $DEFAULT_LR $BETA
    done
}

# ==========================================
# Run Selected Groups
# ==========================================
case $GROUP in
    all)
        run_baseline
        run_capacity_group
        run_learning_group
        run_regularization_group
        ;;
    baseline)
        run_baseline
        ;;
    capacity)
        run_capacity_group
        ;;
    learning)
        run_learning_group
        ;;
    regularization)
        run_regularization_group
        ;;
    *)
        echo "Unknown group: $GROUP"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Stage 0 (VAE) Hyperparameter Analysis Complete!"
echo "=========================================="
echo "Results: $LOG_FILE"
echo ""
echo "To sort by Mean Error:"
echo "  sort -t',' -k9 -n $LOG_FILE | head"
echo "=========================================="
