export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
python3 ./src/experiments/CSDI.py \
   --dataset_type="Electricity" \
   --device="cuda" \
   --batch_size=8 \
   --horizon=1 \
   --layers=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'

