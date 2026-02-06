#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks 1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:A30:4

srun accelerate launch train_grpo_colocate.py


