# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import torch
from reward_funcs import (
    schema_linking_reward,
    query_ngram_comparison_reward,
    syntax_check_reward,
    comprehensive_execution_reward_func,
    subset_match_reward_func,
    execution_exact_match_reward_func,
)

TRAIN_PATH = 'dataset/spider_data/preprocessed/preprocessed_train_spider.json'
TEST_PATH = 'dataset/spider_data/preprocessed/preprocessed_test_spider.json'
MODEL_OUTPUT_PATH = 'base_model'
RUN_NAME = 'All Rewards Qwen3-.6B'

data_files = {'train': TRAIN_PATH, 'test': TEST_PATH}
train_dataset = load_dataset("json", data_files=data_files, split='train')
eval_dataset = load_dataset("json", data_files=data_files, split='test')


class EvalEvery100StepsCallback(TrainerCallback):
    """
    Runs trainer.evaluate() every `eval_steps` training steps.
    Temporarily swaps in eval-specific reward functions during the evaluate()
    call, then restores the original training reward functions afterward.
    """

    def __init__(self, eval_dataset, eval_reward_funcs, eval_steps: int = 100):
        self.eval_dataset = eval_dataset
        self.eval_reward_funcs = eval_reward_funcs
        self.eval_steps = eval_steps

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            print(f"\n[EvalCallback] Running evaluation at step {state.global_step}...")
            original_reward_funcs = self.trainer.reward_funcs
            original_reward_processing_classes = self.trainer.reward_processing_classes
            original_reward_func_names = self.trainer.reward_func_names
            original_reward_weights = self.trainer.reward_weights

            self.trainer.reward_funcs = self.eval_reward_funcs
            self.trainer.reward_processing_classes = [None] * len(self.eval_reward_funcs)
            self.trainer.reward_func_names = [f.__name__ for f in self.eval_reward_funcs]
            self.trainer.reward_weights = torch.ones(len(self.eval_reward_funcs))

            try:
                metrics = self.trainer.evaluate(self.eval_dataset)
                print(f"[EvalCallback] Step {state.global_step} metrics: {metrics}")
            finally:
                self.trainer.reward_funcs = original_reward_funcs
                self.trainer.reward_processing_classes = original_reward_processing_classes
                self.trainer.reward_func_names = original_reward_func_names
                self.trainer.reward_weights = original_reward_weights


training_args = GRPOConfig(
    use_vllm=True,
    max_completion_length=512,
    output_dir=MODEL_OUTPUT_PATH,
    save_strategy="no",
    vllm_mode='colocate',
    report_to='wandb',
    run_name=RUN_NAME,
    per_device_train_batch_size=6,
    eval_strategy="no"
)

eval_reward_funcs = [
    comprehensive_execution_reward_func,
    subset_match_reward_func,
    execution_exact_match_reward_func,
]

eval_callback = EvalEvery100StepsCallback(
    eval_dataset=eval_dataset,
    eval_reward_funcs=eval_reward_funcs,
    eval_steps=100,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[
        schema_linking_reward,
        query_ngram_comparison_reward,
        syntax_check_reward,
        comprehensive_execution_reward_func,
    ],
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    callbacks=[eval_callback],
)

eval_callback.set_trainer(trainer)

print("Start training")
trainer.train()
trainer.save_model(MODEL_OUTPUT_PATH)
