# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from reward_funcs import query_match_reward_func

TRAIN_PATH = './preprocessed_data/train.json'
TEST_PATH = './preprocessed_data/test.json'

data_files = {'train': TRAIN_PATH, 'test': TEST_PATH}
dataset = load_dataset("json", data_files=data_files, split='train')

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode='colocate'
    report_to='wandb',
    run_name='query_match_reward'
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=query_match_reward_func,
    train_dataset=dataset,
    args=training_args
)
print("Start training")
trainer.train()
