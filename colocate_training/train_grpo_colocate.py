# train_grpo.py
from pathlib import Path

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
from reward_funcs import execution_match_reward_func

TRAIN_PATH = REPO_ROOT / "preprocessed_data/train.json"
TEST_PATH = REPO_ROOT / "preprocessed_data/test.json"

data_files = {"train": str(TRAIN_PATH), "test": str(TEST_PATH)}
dataset = load_dataset("json", data_files=data_files, split="train")

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    report_to="wandb",
    run_name="query_match_reward",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=execution_match_reward_func,
    train_dataset=dataset,
    args=training_args
)
print("Start training")
trainer.train()
