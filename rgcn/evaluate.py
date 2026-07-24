# evaluate.py
#!/usr/bin/env python
import os
import sys
import subprocess

# 获取目录
current_dir = os.path.dirname(os.path.abspath(__file__))
imputation_dir = os.path.dirname(current_dir)
data_processing_dir = os.path.join(imputation_dir, 'data_processing')

# 构建命令 
pred_dir = os.path.join(current_dir, 'output', 'preds')
postprocess_script = os.path.join(data_processing_dir, 'postprocess_perseg_aligntime.py')

# 从命令行参数获取target_basin
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--target_basin', type=str, default=None, help='Target basin for metrics logging')
args = parser.parse_args()

target_basin = args.target_basin
metrics_log = os.path.join(current_dir, 'basin_metrics_log.csv') if target_basin else None

# 运行评估
cmd = [
    sys.executable,  # 使用当前Python解释器
    postprocess_script,
    '--pred_dir', pred_dir,
    '--model_name', 'RGCN',
    '--partition', 'tst'
]

# 如果指定了target_basin，添加相关参数
if target_basin:
    cmd.extend(['--target_basin', target_basin])
    if metrics_log:
        cmd.extend(['--metrics_log', metrics_log])

print("Running RGCN evaluation...")
subprocess.run(cmd)
