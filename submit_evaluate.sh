#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --time=08:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8      
#SBATCH --mem=32G              
#SBATCH --gres=gpu:A30:1

module load cuda/12.2.0  

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache

python -u evaluate_grpo_colocate.py