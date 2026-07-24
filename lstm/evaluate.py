# evaluate.py
#!/usr/bin/env python
import os
import sys
import subprocess

# Get directories
current_dir = os.path.dirname(os.path.abspath(__file__))
imputation_dir = os.path.dirname(current_dir)
data_processing_dir = os.path.join(imputation_dir, 'data_processing')

# Build command
pred_dir = os.path.join(current_dir, 'output', 'preds')
postprocess_script = os.path.join(data_processing_dir, 'postprocess_perseg_aligntime.py')

# Get target_basin from command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target_basin', type=str, default=None, help='Target basin for metrics logging')
args = parser.parse_args()

target_basin = args.target_basin
metrics_log = os.path.join(current_dir, 'basin_metrics_log.csv') if target_basin else None

# Run evaluation
cmd = [
    sys.executable,  # use current Python interpreter
    postprocess_script,
    '--pred_dir', pred_dir,
    '--model_name', 'RGCN',
    '--partition', 'tst'
]

# If target_basin is specified, add related arguments
if target_basin:
    cmd.extend(['--target_basin', target_basin])
    if metrics_log:
        cmd.extend(['--metrics_log', metrics_log])

print("Running RGCN evaluation...")
subprocess.run(cmd)
