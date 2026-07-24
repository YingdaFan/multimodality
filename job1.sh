#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:h100-80:1
#SBATCH -N 1
#SBATCH -o output.log

cd $SLURM_SUBMIT_DIR
SI_SIGMA_INT=0 SI_EPS_INFERENCE=0 bash run_si_gx_enc.sh sical 500 3 33 