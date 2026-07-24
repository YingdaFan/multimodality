#!/bin/bash
# Imputation Joint LSTM+Diffusion (Hourly, no-mask, single-run)
#
# Single-run counterpart of run_gx_enc_joint_hourly.sh — does NOT do K-fold
# cross-validation. Uses DUMMY_BASIN for stage 1, so:
#   - No basin gets masked from training y_obs
#   - LSTM trains on ALL 618 basins with their actual observed Y (NaN-aware)
#   - VAE substitution for masked basins is a no-op
#   - Stage 2 (joint diffusion) loads the LSTM into its compute graph and
#     fine-tunes both networks end-to-end
#
# Compared to run_imputationforecast.sh:
#   - This file ONLY does Phase 1 (joint imputation). It does NOT do
#     bridge prepare or forecast. Add Phase 2/3 yourself if you need
#     end-to-end forecast.
#   - Stage 2 here uses the JOINT pipeline (run_gx_enc_joint_stage2.sh)
#     instead of the standard pipeline (run_gx_enc_stage2.sh).
#
# Env vars:
#   CUDA_VISIBLE_DEVICES - which GPU index to use (e.g. 0 or 1; default: first visible)
#   LSTM_LR            - LSTM fine-tune LR for joint stage (default: 1e-5)
#   LSTM_HIDDEN_DIM    - LSTM hidden dim (default: 20)
#
# Usage:
#   bash run_imputation_joint_hourly.sh
#   bash run_imputation_joint_hourly.sh --skip_lstm    # only run joint stage 2
#                                                     # (assumes Stage 1 done)
#
# Recommended (long-running, 4-6 hours):
#   nohup CUDA_VISIBLE_DEVICES=1 bash run_imputation_joint_hourly.sh \
#         > /tmp/joint_hourly.log 2>&1 &

set -e

# ---- Args ----
SKIP_LSTM=false
for arg in "$@"; do
    case $arg in
        --skip_lstm)  SKIP_LSTM=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LSTM_DIR="$SCRIPT_DIR/lstm"
DIFFUSION_DIR="$SCRIPT_DIR/diffusion"
DIFFUSION_SCRIPTS_DIR="$DIFFUSION_DIR/scripts/CAMELS"

# ---- Hourly-specific environment ----
export PREPROCESS_SCRIPT="preprocess_perseg_aligntime_camelsh.py"
export DIFFUSION_PRED_LEN=168
export DIFFUSION_WINDOWS=168

# ---- Banner ----
echo "=========================================="
echo "Imputation Joint LSTM+Diffusion (HOURLY, no-mask)"
echo "=========================================="
echo "Dataset:       camelsh_global.parquet (130 TRB + 488 TRB-like = 618 basins)"
echo "Preprocess:    $PREPROCESS_SCRIPT"
echo "Diffusion:     pred_len=$DIFFUSION_PRED_LEN windows=$DIFFUSION_WINDOWS"
echo "GPU:           ${CUDA_VISIBLE_DEVICES:-all visible}"
echo "LSTM joint LR: ${LSTM_LR:-1e-5}"
echo "Mask:          NONE (DUMMY_BASIN — full data, no spatial CV)"
echo "=========================================="
echo ""

# ---- Phase 1: LSTM (Stage 1, no masking) ----
if [ "$SKIP_LSTM" = false ]; then
    echo "==== Stage 1: LSTM training (no basin masking) ===="
    bash "$LSTM_DIR/run_camels_perstd_stage1_raw.sh" DUMMY_BASIN
fi

# ---- Phase 2: Joint Diffusion Stage 2 ----
echo ""
echo "==== Stage 2: Joint LSTM + Diffusion (Pure Encoder) ===="
bash "$DIFFUSION_SCRIPTS_DIR/run_gx_enc_joint_stage2.sh" diffcal DUMMY_BASIN

echo ""
echo "=========================================="
echo "Joint hourly imputation complete."
echo "  LSTM weights:         $LSTM_DIR/output/finetuned_weights.pth"
echo "  LSTM metrics:         $LSTM_DIR/output/"
echo "  Diffusion metrics:    $DIFFUSION_DIR/output/"
echo "  Predictions:          $DIFFUSION_DIR/output/pred/"
echo "=========================================="
