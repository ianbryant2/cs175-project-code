from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
import argparse


from reward_funcs import comprehensive_execution_reward_func, subset_match_reward_func, execution_exact_match_reward_func


MAX_WORKERS = 16  # Adjust based on CPU cores (matches your cpus-per-task)
    
def evaluate(model_path, test_path, run_name):
    dataset = load_dataset("json", data_files={'train': test_path}, split='train')

    # Evaluating config
    training_args = GRPOConfig(
        output_dir="eval_output",
        max_completion_length=512,     
        use_vllm=True,
        vllm_mode='colocate', # Ensures vLLM shares the GPU with the main process
        report_to='wandb',
        run_name=run_name,
        per_device_eval_batch_size=16, # Default is often 8, bumping this is critical
        
        dataloader_num_workers=8,
    )

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=[comprehensive_execution_reward_func, subset_match_reward_func, execution_exact_match_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting evaluation on saved model...")
    # Passing the dataset directly to evaluate
    metrics = trainer.evaluate(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, help='Model Path', default=None)
    parser.add_argument('--run-name', type=str, help='Run Name', default=None)
    parser.add_argument('--test-path', type=str, default=None)
    args = parser.parse_args()

    evaluate(args.model_path, args.test_path, args.run_name)