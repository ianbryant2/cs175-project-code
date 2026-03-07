#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:2

module load cuda/12.2.0  

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="CS 175 Project LLM"
export WANDB_API_KEY="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"
export DS_SKIP_CUDA_CHECK=1
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export CUDA_HOME=$CUDA_DIR
export PYTHONUNBUFFERED=1



srun accelerate launch --num_processes=1 evaluate_grpo.py