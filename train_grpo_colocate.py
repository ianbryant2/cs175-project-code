# train_grpo.py
from pathlib import Path
from datasets import load_dataset, load_from_disk
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
import torch
import argparse

# Import the individual rewards for Eval, and the Scheduled class for Training
from reward_funcs import (
    ScheduledReward,
    comprehensive_execution_reward_func,
    subset_match_reward_func,
    execution_exact_match_reward_func,
)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Model name to use for chat template', default=None)
parser.add_argument('--train-path', type=str, default='dataset/spider_data/preprocessed/preprocessed_train_spider.json')
parser.add_argument('--test-path', type=str, default='dataset/spider_data/preprocessed/preprocessed_test_spider.json')
args = parser.parse_args()

TRAIN_PATH = args.train_path
TEST_PATH = args.test_path
CACHE_DIR = 'dataset/spider_data/preprocessed/cached'
MODEL_NAME = args.model
RUN_NAME = f'All Reward Funcs'
MODEL_OUTPUT_PATH = 'base_model'


def load_or_cache(train_path, test_path, cache_dir):
    """
    On first run: loads from JSON and saves an Arrow cache to disk.
    On subsequent runs: loads directly from the Arrow cache (much faster).
    """
    cache = Path(cache_dir)
    train_cache = cache / 'train'
    test_cache = cache / 'test'

    if train_cache.exists() and test_cache.exists():
        print("Loading datasets from Arrow cache...")
        train_ds = load_from_disk(str(train_cache))
        test_ds = load_from_disk(str(test_cache))
    else:
        print("Cache not found — loading from JSON and building cache...")
        cache.mkdir(parents=True, exist_ok=True)
        data_files = {'train': train_path, 'test': test_path}
        train_ds = load_dataset("json", data_files=data_files, split='train')
        test_ds = load_dataset("json", data_files=data_files, split='test')
        train_ds.save_to_disk(str(train_cache))
        test_ds.save_to_disk(str(test_cache))
        print("Cache saved.")

    return train_ds, test_ds


train_dataset, eval_dataset = load_or_cache(TRAIN_PATH, TEST_PATH, CACHE_DIR)


class EvalCallback(TrainerCallback):
    """
    Runs trainer.evaluate() every `eval_steps` training steps.
    Temporarily swaps in eval-specific reward functions during the evaluate()
    call, then restores the original training reward functions afterward.
    """

    def __init__(self, eval_dataset, eval_reward_funcs, eval_steps: int = 512):
        self.eval_dataset = eval_dataset
        self.eval_reward_funcs = eval_reward_funcs
        self.eval_steps = eval_steps
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n[EvalCallback] Running evaluation at step {state.global_step}...")

            # Select a small random subset for evaluation speed
            eval_subset = self.eval_dataset.shuffle(seed=state.global_step).select(range(512))

            # Backup original training configuration
            original_reward_funcs = self.trainer.reward_funcs
            original_reward_processing_classes = self.trainer.reward_processing_classes
            original_reward_func_names = self.trainer.reward_func_names
            original_reward_weights = self.trainer.reward_weights

            # Swap in the static evaluation rewards
            self.trainer.reward_funcs = self.eval_reward_funcs
            self.trainer.reward_processing_classes = [None] * len(self.eval_reward_funcs)
            self.trainer.reward_func_names = [f.__name__ for f in self.eval_reward_funcs]
            self.trainer.reward_weights = torch.ones(len(self.eval_reward_funcs))

            try:
                metrics = self.trainer.evaluate(eval_subset)
                print(f"[EvalCallback] Step {state.global_step} metrics: {metrics}")
            finally:
                # Restore the scheduled training reward
                self.trainer.reward_funcs = original_reward_funcs
                self.trainer.reward_processing_classes = original_reward_processing_classes
                self.trainer.reward_func_names = original_reward_func_names
                self.trainer.reward_weights = original_reward_weights


training_args = GRPOConfig(
    use_vllm=True,
    max_completion_length=512,
    vllm_max_model_length=16384,
    output_dir=MODEL_OUTPUT_PATH,
    save_strategy="no",
    vllm_mode='colocate',
    report_to='wandb',
    run_name=f'{RUN_NAME} | Model: {MODEL_NAME}',
    per_device_train_batch_size=4,
    eval_strategy="no",
)

# 1. Initialize the scheduled reward OBJECT
scheduled_reward_func = ScheduledReward()

# 2. Define the static evaluation rewards
eval_reward_funcs = [
    comprehensive_execution_reward_func,
    subset_match_reward_func,
    execution_exact_match_reward_func,
]

eval_callback = EvalCallback(
    eval_dataset=eval_dataset,
    eval_reward_funcs=eval_reward_funcs,
    eval_steps=512,
)

trainer = GRPOTrainer(
    model=MODEL_NAME,
    # Pass the object instance, not the class
    reward_funcs=scheduled_reward_func,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    callbacks=[eval_callback],
)

# 3. CRITICAL: Inject the trainer into the reward object so it can track progress
scheduled_reward_func.set_trainer(trainer)
eval_callback.set_trainer(trainer)

print("Start training")
trainer.train()
trainer.save_model(MODEL_OUTPUT_PATH)