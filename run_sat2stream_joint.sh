#!/bin/bash
# Sat2Stream JOINT variant: stage-1 pretraining, then stage-2 fine-tunes the
# LSTM + pixel-set encoder INSIDE the diffusion graph (gradients flow from
# the diffusion loss into the prior and the encoder; small LSTM LR).
# The live embedding enters both the prior input and the calibrator's
# condition on every forward pass -- no frozen-embedding bridge (augment is
# not used in this pipeline).
#
# Evaluation protocol: same as run_sat2stream.sh (TRB-101 targets from
# data_processing/trb_targets_eval.txt, masked in stage 1, evaluated in
# stage 2).
#
# Configuration:
#   lstm/config.yml use_smap must be true (multimodal stage 1)
#   SMAP_ENC_ATTRS=1   optional attribute-conditioned encoder
#   LSTM_LR            joint fine-tune LR for LSTM+encoder (default 1e-5)
set -e
cd "$(dirname "$0")"
BASINS=$(cat data_processing/trb_targets_eval.txt)
rm -f lstm/output/basin_metrics_log*.csv lstm/output/vae_statistics_log.csv \
      lstm/output/basin_metrics_combined.csv diffusion/output/basin_metrics_log*.csv
bash lstm/run_camels_perstd_stage1_raw.sh $BASINS
bash diffusion/scripts/CAMELS/run_joint_smap_stage2.sh diffcal $BASINS
echo "Sat2Stream JOINT run complete."
