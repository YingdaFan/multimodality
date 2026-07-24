#!/bin/bash

export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/NsDiff_CAMELS.py \
   --dataset_type="CAMELS" \
   --npz_path="../data_processing/data/prepped.npz" \
   --device="cuda" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=365 \
   --windows=365 \
   --load_pretrain=False \
   --epochs=300 \
   --patience=10 \
   --lr=0.001 \
   runs --seeds='[1]'
