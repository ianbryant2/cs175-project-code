# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from reward_funcs import schema_linking_reward, query_ngram_comparison_reward, syntax_check_reward

TRAIN_PATH = '../dataset/spider_data/preprocessed/preprocessed_train_spider.json'
TEST_PATH = '../dataset/spider_data/preprocessed/preprocessed_test_spider.json'

data_files = {'train': TRAIN_PATH, 'test': TEST_PATH}
dataset = load_dataset("json", data_files=data_files, split='train')

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode='colocate',
    report_to='wandb',
    run_name='schema, syntax, and ngram rewards',
    per_device_train_batch_size=2
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[schema_linking_reward, query_ngram_comparison_reward, syntax_check_reward],
    train_dataset=dataset,
    args=training_args
)
print("Start training")
trainer.train()
