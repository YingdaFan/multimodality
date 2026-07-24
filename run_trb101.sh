#!/bin/bash
# Zero-shot reconstruction of 101 TRB basins (SMAP-era coverage > 50%).
# Masks all of them in Stage 1, evaluates Stage 2 on them. No fold loop.
set -e
cd "$(dirname "$0")"
BASINS=$(cat data_processing/basins_trb_cov50.txt)
rm -f lstm/output/basin_metrics_log*.csv lstm/output/vae_statistics_log.csv \
      lstm/output/basin_metrics_combined.csv diffusion/output/basin_metrics_log*.csv
bash lstm/run_camels_perstd_stage1_raw.sh $BASINS
bash diffusion/scripts/CAMELS/run_gx_enc_stage2.sh diffcal $BASINS
echo "TRB-101 run complete."
