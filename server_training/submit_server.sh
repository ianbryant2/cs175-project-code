#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --time=36:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:V100:4

# Pick nodes from allocation
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER_NODE=${NODES[0]}
TRAIN_NODE=${NODES[1]}

INFO_FILE="$SLURM_SUBMIT_DIR/server_info.txt"
rm -f "$INFO_FILE"

echo "Starting vLLM server on $SERVER_NODE ..."
srun --nodes=1 --ntasks=1 --exclusive --nodelist="$SERVER_NODE" bash -lc "
  echo $SERVER_NODE > '$INFO_FILE'
  conda activate RL
  trl vllm-serve \
    --model 'Qwen/Qwen2-0.5B-Instruct' \
    --host 0.0.0.0 --port 8000 > server.out 2>&1
" &
SERVER_STEP_PID=$!

# Wait until the server node info exists
while [ ! -s "$INFO_FILE" ]; do sleep 0.2; done
SERVER_HOST=$(cat "$INFO_FILE")

echo "Waiting for server health on $SERVER_HOST..."
until curl -sf "http://$SERVER_HOST:8000/health/" >/dev/null 2>&1; do
  sleep 1
done
echo "Server has started"

echo "Launching training on $TRAIN_NODE ..."
srun --nodes=1 --ntasks=1 --exclusive --nodelist="$TRAIN_NODE" bash -lc "
  export VLLM_BASE_URL='http://$SERVER_HOST:8000'
  conda activate RL
  accelerate launch train_grpo_server.py
"

echo "Training finished; stopping server"
kill $SERVER_STEP_PID

