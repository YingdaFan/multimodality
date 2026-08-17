#!/bin/bash
# TRB-101 multimodal run, mean-pool readout (the original scheme, pinned).
# Ablation row 1: equal-weight scatter mean over each basin's pixel set.
set -e
cd "$(dirname "$0")"
export USE_SMAP=1
export SMAP_READOUT=mean
export SMAP_QUERY_CTX=
bash run_sat2stream_joint.sh
