#!/bin/bash
# Imputation → Forecasting 完整流水线 (Hourly CAMELS-H, 130 basins)
# 单次运行，不 mask，不 K-fold
#
# Usage:
#   bash run_full_pipeline.sh
#   bash run_full_pipeline.sh --skip_imputation   # 跳过 imputation（已有 trn.npy）
#   bash run_full_pipeline.sh --skip_forecast      # 只跑 imputation

set -e

SKIP_IMPUTATION=false
SKIP_FORECAST=false
for arg in "$@"; do
    case $arg in
        --skip_imputation) SKIP_IMPUTATION=true ;;
        --skip_forecast)   SKIP_FORECAST=true ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PREPROCESS_SCRIPT="preprocess_perseg_aligntime_camelsh.py"

# ---- Phase 1: Imputation (LSTM → Diffusion calibration) ----
if [ "$SKIP_IMPUTATION" = false ]; then
    echo "==== Phase 1: Imputation ===="
    # LSTM 阶段：包含 preprocess，产出 data/prepped.npz（窗口化）
    bash "$SCRIPT_DIR/lstm/run_camels_perstd_stage1_raw.sh" DUMMY_BASIN

    # Imputation diffusion 的 seq_len 由 NPZ 形状决定（单一真源）
    # 因为 imputation NPZ 是窗口化的 (n_win*n_bas, seq_len, feat)，窗口已固化在数据里
    SEQ_LEN=$(python3 -c "import numpy as np; print(np.load('$SCRIPT_DIR/data_processing/data/prepped.npz')['x_trn'].shape[1])")
    echo "Derived seq_len from NPZ: $SEQ_LEN"
    export DIFFUSION_PRED_LEN=$SEQ_LEN
    export DIFFUSION_WINDOWS=$SEQ_LEN

    bash "$SCRIPT_DIR/diffusion/scripts/CAMELS/run_gx_enc_stage2.sh" diffcal DUMMY_BASIN
fi

# ---- Phase 2: Bridge (extract metadata before forecast overwrites NPZ) ----
echo "==== Phase 2: Bridge (prepare metadata) ===="
cd "$SCRIPT_DIR/data_processing"
python3 fill_forecast_npz.py prepare --imputation_npz data/prepped.npz

# ---- Phase 3: Forecasting (preprocess → inject → train → postprocess) ----
# 注意：forecast 的 seq_len 由 run_forecast.sh 的 CLI 参数决定（--windows / --pred_len），
# 与 imputation 的 SEQ_LEN 无关，不需要传递。
if [ "$SKIP_FORECAST" = false ]; then
    echo "==== Phase 3: Forecasting ===="
    cd "$SCRIPT_DIR"
    bash run_forecast.sh
fi

echo "==== Done ===="
