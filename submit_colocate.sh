#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:A30:4

module load cuda/12.2.0  

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="CS 175 Project LLM"
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export CUDA_HOME=$CUDA_DIR

MODEL=${1:?"Error: model name required. Usage: sbatch submit.sh <model_name>"}
SKIP_REPROCESS=false
for arg in "$@"; do
    if [ "$arg" = "-r" ]; then
        SKIP_REPROCESS=true
    fi
done

cd dataset

if [ "$SKIP_REPROCESS" = false ]; then
    echo "Preprocessing datasets..."
    python3 preprocess.py -a -m "$MODEL" -o preprocessed_train_spider.json -d train
    python3 preprocess.py -m "$MODEL" -o preprocessed_test_spider.json -d test
else
    echo "Skipping preprocessing (-r flag set)."
fi

cd ..

srun accelerate launch --config_file accelerate_config.yaml train_grpo_colocate.py --model "$MODEL"