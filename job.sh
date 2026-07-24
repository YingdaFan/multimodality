#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH -o output.log

cd $SLURM_SUBMIT_DIR
bash run_gx_enc.sh diffcal 500 3 33 