from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig


from reward_funcs import comprehensive_execution_reward_func, subset_match_reward_func, execution_exact_match_reward_func

MODEL_PATH = "base_model"
TEST_DATA_PATH = "dataset/spider_data/preprocessed/preprocessed_test_spider.json"
DATABASE_BASE_DIRECTORY = "dataset/spider_data/test_database"
MAX_WORKERS = 16  # Adjust based on CPU cores (matches your cpus-per-task)


    
def evaluate():
    dataset = load_dataset("json", data_files={'train': TEST_DATA_PATH}, split='train')

    # Evaluating config
    training_args = GRPOConfig(
        output_dir="eval_output",
        max_completion_length=512,     
        use_vllm=True,
        vllm_mode='colocate', # Ensures vLLM shares the GPU with the main process
        report_to='wandb',
        run_name='Evaluate baseline',
        per_device_eval_batch_size=16, # Default is often 8, bumping this is critical
        
        dataloader_num_workers=8,
    )

    trainer = GRPOTrainer(
        model=MODEL_PATH,
        reward_funcs=[comprehensive_execution_reward_func, subset_match_reward_func, execution_exact_match_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting evaluation on saved model...")
    # Passing the dataset directly to evaluate
    metrics = trainer.evaluate(dataset)


if __name__ == "__main__":
    evaluate()