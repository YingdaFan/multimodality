#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100|l40s|v100-32"
#SBATCH -N 1
#SBATCH --cpus-per-task=5
#SBATCH --mem=22000M
#SBATCH -o output.log

cd $SLURM_SUBMIT_DIR

bash run_forecast.sh --model futuretst --no_imputed

# bash run_forecast.sh --model futuretst
# bash run_forecast.sh --model nsdiff --no_imputed
# bash run_forecast.sh --model nsdiff
# bash run_forecast.sh --model patchtst --no_imputed
# bash run_forecast.sh --model patchtst
# bash run_forecast.sh --model lstm --no_imputed
# bash run_forecast.sh --model lstm
# bash run_forecast.sh --model dlinear --no_imputed
# bash run_forecast.sh --model informer --no_imputed
# bash run_forecast.sh --model itransformer --no_imputed

# 调超参示例：
# bash run_forecast.sh --model futuretst --no_imputed --windows 168 --pred_len 18 --epochs 200 --lr 0.001
