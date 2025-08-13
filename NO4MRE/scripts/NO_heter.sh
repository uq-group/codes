#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4
#SBATCH --time=6:00:00
#SBATCH --account=bbqg-delta-gpu
#SBATCH --job-name=no_heter
#SBATCH --output="scripts/logs/no_heter.out"

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
###SBATCH --gpu-bind=none     # <- or closest

module purge # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)

module load openmpi

source ~/.bashrc  # Ensure conda is set up
conda activate base  # Replace 'my_env' with your actual environment name

module list  # job documentation and metadata

cd /u/wzhong/PhD/Github/NO4mre/

echo "job is starting on `hostname`"

which python3
conda list pytorch

python3 NO.py --model='FNO' --phase='train' --data='heter'
python3 NO.py --model='WNO' --phase='train' --data='heter'
python3 NO.py --model='Unet' --phase='train' --data='heter'
python3 NO.py --model='UFNO' --phase='train' --data='heter'