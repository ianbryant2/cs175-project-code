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
export WANDB_PROJECT="CS 175 Project LLM Eval"
export WANDB_API_KEY="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"
export DS_SKIP_CUDA_CHECK=1
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache
export CUDA_HOME=$CUDA_DIR
export PYTHONUNBUFFERED=1

MODEL_PATH=${1:?"Usage: sbatch submit_evaluate.sh <model_path> <split_dir>"}
SPLIT_DIR=${2:?"Usage: sbatch submit_evaluate.sh <model_path> <split_dir>"}
NUM_SPLITS=$(ls "${SPLIT_DIR}"/*.json 2>/dev/null | wc -l)

for i in $(seq 1 "$NUM_SPLITS"); do
    TEST_PATH="${SPLIT_DIR}/${i}.json"
    RUN_NAME="Eval | Model: ${MODEL_PATH} | Split ${i}"

    if [ ! -f "$TEST_PATH" ]; then
        echo "Missing $TEST_PATH, skipping split $i"
        continue
    fi

    echo "Evaluating split $i with $TEST_PATH"
    srun accelerate launch --num_processes=1 evaluate_grpo.py \
        --model-path "$MODEL_PATH" \
        --test-path "$TEST_PATH" \
        --run-name "$RUN_NAME"
done
