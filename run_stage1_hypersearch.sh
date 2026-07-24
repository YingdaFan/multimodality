#!/bin/bash
# Stage 1 (LSTM) Hyperparameter Sensitivity Analysis
#
# Design inspired by TimeDart (ICML 2025):
# - Group experiments by module/component
# - Control variable method: change one parameter at a time
# - Include baseline (default config) for comparison
#
# This script:
# 1. Runs preprocessing ONCE
# 2. Runs grouped LSTM hyperparameter experiments
#
# Usage:
#   bash run_stage1_hypersearch.sh <basin_id>
#   bash run_stage1_hypersearch.sh <basin_id> <group_name>
#   bash run_stage1_hypersearch.sh --skip-preprocess <basin_id> <group_name>
#
# Groups:
#   all        - Run all groups (default)
#   baseline   - Baseline only
#   capacity   - Hidden size
#   learning   - Learning rate
#   regularization - Dropout and weight decay

if [ $# -eq 0 ]; then
    echo "Usage: bash run_stage1_hypersearch.sh [--skip-preprocess] <basin_id> [group_name]"
    echo ""
    echo "Options:"
    echo "  --skip-preprocess  Skip preprocessing (use existing prepped.npz with masked basin)"
    echo ""
    echo "Groups: all, baseline, capacity, learning, regularization"
    echo ""
    echo "Examples:"
    echo "  bash run_stage1_hypersearch.sh 01022500              # Run all groups"
    echo "  bash run_stage1_hypersearch.sh 01022500 capacity     # Run capacity group only"
    echo "  bash run_stage1_hypersearch.sh --skip-preprocess 01022500 learning"
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
echo "Stage 1 (LSTM) Hyperparameter Analysis"
echo "=========================================="
echo "Target Basin: $BASIN_ID"
echo "Experiment Group: $GROUP"
echo ""

# ==========================================
# Preprocessing: Run ONCE
# ==========================================
if [ "$SKIP_PREPROCESS" = true ]; then
    echo "=========================================="
    echo "Preprocessing: SKIPPED (--skip-preprocess)"
    echo "=========================================="

    PREPPED_NPZ="$DATA_DIR/data/prepped.npz"
    if [ ! -f "$PREPPED_NPZ" ]; then
        echo "ERROR: prepped.npz not found at $PREPPED_NPZ"
        exit 1
    fi
    echo "Using existing: $PREPPED_NPZ"
else
    echo "=========================================="
    echo "Preprocessing: Prepare data and mask basin"
    echo "=========================================="

    cd "$DATA_DIR"

    echo "[Step 1/3] Preprocessing data..."
    python preprocess_perseg_aligntime_camels.py

    echo "[Step 2/3] Masking basin in training data..."
    python modify_basin_to_nan_allmask.py $BASIN_ID

    echo "[Step 3/3] Applying VAE correction..."
    python apply_vae.py $BASIN_ID --script_dir "$LSTM_DIR"

    echo "Preprocessing complete."
fi
echo ""

# ==========================================
# Output Directory
# ==========================================
HYPERSEARCH_OUTPUT="$LSTM_DIR/output/hypersearch/${BASIN_ID}"
mkdir -p "$HYPERSEARCH_OUTPUT"

# Backup original config
CONFIG_FILE="$LSTM_DIR/config.yml"
CONFIG_BACKUP="$HYPERSEARCH_OUTPUT/config_backup.yml"
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

# Log file
LOG_FILE="$HYPERSEARCH_OUTPUT/experiment_results.csv"

# Initialize log file
if [[ ! -f "$LOG_FILE" ]] || [[ "$GROUP" == "all" && "$SKIP_PREPROCESS" = false ]]; then
    echo "group,exp_name,hidden_size,dropout,lr,weight_decay,epochs,early_stopping,nse,kge,rmse,mae,r2,pbias,status" > "$LOG_FILE"
    echo "Initialized new log file: $LOG_FILE"
else
    echo "Appending to existing log file: $LOG_FILE"
fi

# ==========================================
# Default (Baseline) Configuration
# ==========================================
DEFAULT_HIDDEN_SIZE=20
DEFAULT_DROPOUT=0.2
DEFAULT_LR=0.001
DEFAULT_WEIGHT_DECAY=0.001
DEFAULT_EPOCHS=150
DEFAULT_EARLY_STOPPING=20

# ==========================================
# Helper: Generate config.yml
# ==========================================
generate_config() {
    local HIDDEN_SIZE=$1
    local DROPOUT=$2
    local LR=$3
    local WEIGHT_DECAY=$4
    local EPOCHS=$5
    local EARLY_STOPPING=$6

    cat > "$CONFIG_FILE" << EOF
hidden_size: ${HIDDEN_SIZE}
dropout: ${DROPOUT}
seed: 42
ft_epochs: ${EPOCHS}
pt_epochs: 50
early_stopping: ${EARLY_STOPPING}
pretrain_learning_rate: 0.005
finetune_learning_rate: ${LR}
weight_decay: ${WEIGHT_DECAY}
trn_offset: 1.0
tst_val_offset: 1.0
out_dir: output
EOF
}

# ==========================================
# Helper: Run Single Experiment
# ==========================================
run_experiment() {
    local GROUP_NAME=$1
    local EXP_NAME=$2
    local HIDDEN_SIZE=$3
    local DROPOUT=$4
    local LR=$5
    local WEIGHT_DECAY=$6
    local EPOCHS=$7
    local EARLY_STOPPING=$8

    echo ""
    echo "----------------------------------------"
    echo "[$GROUP_NAME] $EXP_NAME"
    echo "----------------------------------------"
    echo "  hidden_size=$HIDDEN_SIZE, dropout=$DROPOUT"
    echo "  lr=$LR, weight_decay=$WEIGHT_DECAY"
    echo "  epochs=$EPOCHS, early_stopping=$EARLY_STOPPING"

    # Create experiment output directory
    EXP_OUTPUT="$HYPERSEARCH_OUTPUT/${GROUP_NAME}/${EXP_NAME}"
    mkdir -p "$EXP_OUTPUT"

    # Generate config
    generate_config $HIDDEN_SIZE $DROPOUT $LR $WEIGHT_DECAY $EPOCHS $EARLY_STOPPING

    # Run LSTM training
    cd "$LSTM_DIR"
    python base.py 2>&1 | tee "$EXP_OUTPUT/train.log"
    TRAIN_STATUS=$?

    # Initialize metrics
    NSE="NA"
    KGE="NA"
    RMSE="NA"
    MAE="NA"
    R2="NA"
    PBIAS="NA"
    STATUS="failed"

    if [ $TRAIN_STATUS -eq 0 ]; then
        # Run evaluation
        echo ""
        echo "Running evaluation..."

        cd "$DATA_DIR"
        EVAL_OUTPUT=$(python postprocess_perseg_aligntime.py \
            --pred_dir "$LSTM_DIR/output/preds" \
            --model_name "lstm_hypersearch" \
            --partition trn \
            --target_basin "$BASIN_ID" \
            --use_vae_stats 2>&1)

        echo "$EVAL_OUTPUT" > "$EXP_OUTPUT/eval.log"

        # Extract metrics
        NSE=$(echo "$EVAL_OUTPUT" | grep -oP "NSE: \K[-0-9.]+" | head -1)
        KGE=$(echo "$EVAL_OUTPUT" | grep -oP "KGE: \K[-0-9.]+" | head -1)
        RMSE=$(echo "$EVAL_OUTPUT" | grep -oP "RMSE: \K[-0-9.]+" | head -1)
        MAE=$(echo "$EVAL_OUTPUT" | grep -oP "MAE: \K[-0-9.]+" | head -1)
        R2=$(echo "$EVAL_OUTPUT" | grep -oP "R²: \K[-0-9.]+" | head -1)
        PBIAS=$(echo "$EVAL_OUTPUT" | grep -oP "PBIAS: \K[-0-9.]+" | head -1)

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

    fi

    # Log result
    echo "$GROUP_NAME,$EXP_NAME,$HIDDEN_SIZE,$DROPOUT,$LR,$WEIGHT_DECAY,$EPOCHS,$EARLY_STOPPING,$NSE,$KGE,$RMSE,$MAE,$R2,$PBIAS,$STATUS" >> "$LOG_FILE"

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
        $DEFAULT_HIDDEN_SIZE $DEFAULT_DROPOUT $DEFAULT_LR $DEFAULT_WEIGHT_DECAY \
        $DEFAULT_EPOCHS $DEFAULT_EARLY_STOPPING
}

# ==========================================
# Group 1: Model Capacity (Hidden Size)
# ==========================================
run_capacity_group() {
    echo ""
    echo "=========================================="
    echo "Group 1: Model Capacity (Hidden Size)"
    echo "=========================================="
    echo "Testing: hidden_size in {10, 20, 50, 100}"
    echo "Fixed: lr=$DEFAULT_LR, dropout=$DEFAULT_DROPOUT"

    for HIDDEN_SIZE in 10 20 50 100; do
        run_experiment "capacity" "hidden_${HIDDEN_SIZE}" \
            $HIDDEN_SIZE $DEFAULT_DROPOUT $DEFAULT_LR $DEFAULT_WEIGHT_DECAY \
            $DEFAULT_EPOCHS $DEFAULT_EARLY_STOPPING
    done
}

# ==========================================
# Group 2: Learning Rate
# ==========================================
run_learning_group() {
    echo ""
    echo "=========================================="
    echo "Group 2: Learning Rate Analysis"
    echo "=========================================="
    echo "Testing: lr in {0.01, 0.005, 0.001, 0.0005}"
    echo "Fixed: hidden_size=$DEFAULT_HIDDEN_SIZE"

    for LR in 0.01 0.005 0.001 0.0005; do
        run_experiment "learning" "lr_${LR}" \
            $DEFAULT_HIDDEN_SIZE $DEFAULT_DROPOUT $LR $DEFAULT_WEIGHT_DECAY \
            $DEFAULT_EPOCHS $DEFAULT_EARLY_STOPPING
    done
}

# ==========================================
# Group 3: Regularization (Dropout + Weight Decay)
# ==========================================
run_regularization_group() {
    echo ""
    echo "=========================================="
    echo "Group 3: Regularization Analysis"
    echo "=========================================="
    echo "Testing: dropout in {0.0, 0.1, 0.2, 0.3}"
    echo "         weight_decay in {0.0, 0.001, 0.01}"

    # Dropout variations
    for DROPOUT in 0.0 0.1 0.2 0.3; do
        run_experiment "regularization" "dropout_${DROPOUT}" \
            $DEFAULT_HIDDEN_SIZE $DROPOUT $DEFAULT_LR $DEFAULT_WEIGHT_DECAY \
            $DEFAULT_EPOCHS $DEFAULT_EARLY_STOPPING
    done

    # Weight decay variations
    for WD in 0.0 0.001 0.01; do
        run_experiment "regularization" "wd_${WD}" \
            $DEFAULT_HIDDEN_SIZE $DEFAULT_DROPOUT $DEFAULT_LR $WD \
            $DEFAULT_EPOCHS $DEFAULT_EARLY_STOPPING
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
        echo "Available groups: all, baseline, capacity, learning, regularization"
        # Restore original config
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        exit 1
        ;;
esac

# ==========================================
# Restore original config
# ==========================================
cp "$CONFIG_BACKUP" "$CONFIG_FILE"
echo ""
echo "Restored original config.yml"

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Stage 1 Hyperparameter Analysis Complete!"
echo "=========================================="
echo "Basin: $BASIN_ID"
echo "Group: $GROUP"
echo ""
echo "Experiment Summary:"
echo "  - baseline:       1 experiment  (default configuration)"
echo "  - capacity:       4 experiments (hidden_size: 10, 20, 50, 100)"
echo "  - learning:       4 experiments (lr: 0.01, 0.005, 0.001, 0.0005)"
echo "  - regularization: 7 experiments (dropout + weight_decay)"
echo "  - Total:         16 experiments"
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
echo "  sort -t',' -k9 -rn $LOG_FILE | head"
echo "=========================================="
