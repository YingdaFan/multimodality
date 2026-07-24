#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH -o output_midm_cal.log

cd $SLURM_SUBMIT_DIR
conda activate diffusion
bash run_midm_cal_kfold.sh 22 3 22
