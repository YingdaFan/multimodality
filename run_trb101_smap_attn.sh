#!/bin/bash
# TRB-101 multimodal run, attention readout (D4RT-style query decoding).
# The query attends over the basin's own pixel tokens; SMAP_QUERY_CTX picks
# what the query is built from (comma list):
#   grid_size     pixel count + bounding-grid width/height
#   pixel_mean    mean of the pixel tokens (data-derived context)
#   static_attrs  static_enc basin attributes
# Default is the D4RT-faithful row: data-derived context only.
# Examples:
#   bash run_trb101_smap_attn.sh
#   SMAP_QUERY_CTX=grid_size,pixel_mean,static_attrs bash run_trb101_smap_attn.sh
set -e
cd "$(dirname "$0")"
export USE_SMAP=1
export SMAP_READOUT=attn
export SMAP_QUERY_CTX=${SMAP_QUERY_CTX:-grid_size,pixel_mean}
bash run_sat2stream.sh
