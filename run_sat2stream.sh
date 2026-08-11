#!/bin/bash
# Sat2Stream two-stage pipeline: Stage-1 prior (LSTM) + Stage-2 calibration.
#
# Evaluation protocol (fixed for now): zero-shot on the TRB-101 targets --
# data_processing/trb_targets_eval.txt, the 101 Tennessee River Basin gauges
# with >50% streamflow coverage in the SMAP era. All of them are masked in
# Stage 1 and evaluated in Stage 2. No fold loop.
#
# Method configuration is selected via environment variables:
#   lstm/config.yml use_smap  false = no-satellite baseline / true = multimodal
#   SMAP_FILM=1               FiLM modulation of the calibrator (best config)
#   SMAP_ENC_ATTRS=1          attribute-conditioned satellite encoder (16 attrs)
#   SMAP_FILM_TRUST=1         deprecated (trust attrs in the gate; worse)
# Examples:
#   bash run_sat2stream.sh                                    # per config.yml
#   SMAP_FILM=1 bash run_sat2stream.sh                        # + FiLM
#   SMAP_ENC_ATTRS=1 SMAP_FILM=1 bash run_sat2stream.sh      # + attr encoder
set -e
cd "$(dirname "$0")"
BASINS=$(cat data_processing/trb_targets_eval.txt)
rm -f lstm/output/basin_metrics_log*.csv lstm/output/vae_statistics_log.csv \
      lstm/output/basin_metrics_combined.csv diffusion/output/basin_metrics_log*.csv
bash lstm/run_camels_perstd_stage1_raw.sh $BASINS
bash diffusion/scripts/CAMELS/run_gx_enc_stage2.sh diffcal $BASINS
echo "TRB-101 run complete."
