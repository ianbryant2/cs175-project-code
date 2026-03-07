from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from pathlib import Path
import sqlite3
from collections import Counter
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


from reward_funcs import extract_query_from_response

MODEL_PATH = "base_model"
TEST_DATA_PATH = "dataset/spider_data/preprocessed/preprocessed_test_spider.json"
DATABASE_BASE_DIRECTORY = "dataset/spider_data/test_database"
MAX_WORKERS = 16  # Adjust based on CPU cores (matches your cpus-per-task)


# 1. Caching DB paths prevents repeated filesystem checks
@lru_cache(maxsize=None)
def _get_db_path(db_id: str) -> Path:
    base = Path(DATABASE_BASE_DIRECTORY) / db_id
    for ext in (".sqlite", ".db"):
        p = base / f"{db_id}{ext}"
        if p.exists():
            return p
    return Path()


def _safe_execute_sql(db_path, query):
    """
    Helper to execute SQL safely in a separate thread.
    Creates a fresh connection per call to ensure thread safety.
    """
    try:
        # timeout prevents locking issues
        with closing(sqlite3.connect(db_path, timeout=10)) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = [str(r) for r in cur.fetchall()]
            # Extract column names safely
            cols = set([desc[0].lower() for desc in cur.description]) if cur.description else set()
            return rows, cols
    except Exception:
        return None, None

        
def comprehensive_execution_reward_func(completions, query_result, query_result_columns, db_id, **kwargs):
    """Parallelized execute pred_query and calculate row similarity and column similarity scores"""
    def process_single(args):
        complete, gold_rows, gold_column_names, db = args
        try:
            pred_query = extract_query_from_response(complete[0]['content'])
            db_path = _get_db_path(db)
            
            # Execute SQL
            pred_rows, pred_cols = _safe_execute_sql(db_path, pred_query)
            if pred_rows is None: 
                return 0.0

            # Score 1: Structural Accuracy (Column)
            gold_cols = set(gold_column_names) if gold_column_names else set()
            
            if not gold_cols:
                col_score = 0.0
            else:
                intersection = pred_cols.intersection(gold_cols)
                union = pred_cols.union(gold_cols)
                col_score = len(intersection) / len(union) if union else 0.0
            
            # Score 2: Data Accuracy (Row)
            pred_counter = Counter(pred_rows)
            gold_counter = Counter(gold_rows)
                         
            intersection = pred_counter & gold_counter 
            tp = sum(intersection.values()) # True Positives
            
            fp = sum((pred_counter - gold_counter).values()) # Extra rows
            fn = sum((gold_counter - pred_counter).values()) # Missing rows

            if tp == 0:
                row_score = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                row_score = 2 * (precision * recall) / (precision + recall)

            # Final Weighted Score
            return (0.3 * col_score) + (0.7 * row_score)

        except Exception:
            return 0.0

    # Execute efficiently in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(process_single, zip(completions, query_result, query_result_columns, db_id)))
        

def subset_match_reward_func(completions, query_result, db_id, **kwargs):
    """Parallelized execute pred_query and give 1.0 for an extact match of gold_rows and pred_rows and 0.0"""
    def process_single(args):
        complete, gold_rows, db = args
        try:
            pred_query = extract_query_from_response(complete[0]['content'])
            db_path = _get_db_path(db)
            
            pred_rows, _ = _safe_execute_sql(db_path, pred_query)
            if pred_rows is None:
                return 0.0

            pred_set = set(pred_rows)
            gold_set = set(gold_rows)

            if not gold_set:
                return 1.0 if not pred_set else 0.0

            if pred_set == gold_set:
                return 1.0

            # Case: Pred is a Subset of Gold (Missing rows)
            elif pred_set.issubset(gold_set):
                return len(pred_set) / len(gold_set)

            # Case: Gold is a Subset of Pred (Extra rows)
            elif gold_set.issubset(pred_set):
                return len(gold_set) / len(pred_set) if pred_set else 0.0
                
            # Case: Neither (Mixed overlap or disjoint)    
            else:
                inter = len(pred_set.intersection(gold_set))
                union = len(pred_set.union(gold_set))
                return inter / union if union > 0 else 0.0

        except Exception:
            return 0.0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(process_single, zip(completions, query_result, db_id)))


def execution_exact_match_reward_func(completions, query_result, db_id, **kwargs):
    """Parallelized execute pred_query and calculate row similarity and column similarity scores"""
    def process_single(args):
        complete, gold_rows, db = args
        try:
            pred_query = extract_query_from_response(complete[0]['content'])
            db_path = _get_db_path(db)
            
            pred_rows, _ = _safe_execute_sql(db_path, pred_query)
            if pred_rows is None:
                return 0.0
            
            if Counter(pred_rows) == Counter(gold_rows):
                return 1.0
            else:
                return 0.0

        except Exception:
            return 0.0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(process_single, zip(completions, query_result, db_id)))

    
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