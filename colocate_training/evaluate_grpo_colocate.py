import os
import torch
import sqlite3
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from reward_funcs import extract_query_from_response


CHECKPOINT_PATH = "model/"
BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
TEST_DATA_PATH = "../dataset/spider_data/preprocessed/preprocessed_test_spider.json"
DATABASE_BASE_DIRECTORY = "../dataset/spider_data/test_database"


def _get_db_path(db_id: str) -> Path:
    base = Path(DATABASE_BASE_DIRECTORY) / db_id
    for ext in (".sqlite", ".db"):
        p = base / f"{db_id}{ext}"
        if p.exists():
            return p
    return Path()


def _execute_both_queries(gold_sql: str, pred_sql: str, db_id: str) -> tuple:
    """Connects to DB once and executes both queries, returning results."""
    db_path = _get_db_path(db_id)
    if not db_path.exists():
        return False, [], False, []

    pred_status, pred_rows = False, []
    gold_status, gold_rows = False, []

    try:
        # One connection for both tasks
        conn = sqlite3.connect(db_path)        
        cursor = conn.cursor()

        # Execute Gold
        try:
            cursor.execute(gold_sql)
            gold_rows = cursor.fetchall()
            gold_status = True
        except Exception:
            gold_status = False

        # Execute Pred
        try:
            cursor.execute(pred_sql)
            pred_rows = cursor.fetchall()
            pred_status = True
        except Exception:
            pred_status = False

        conn.close()
    except Exception as e:
        print(f"query execution error: {e}")

    return gold_status, gold_rows, pred_status, pred_rows


def column_level_match(pred_rows, gold_rows):
    if not gold_rows: 
        return 0.0
    if not pred_rows: 
        return 0.0

    # Transpose: List of Rows -> List of Columns
    try:
        gold_cols = list(zip(*gold_rows))
        pred_cols = list(zip(*pred_rows))
    except:
        return 0.0

    def process_col(col):
        return sorted([str(x) for x in col])

    g_processed = [process_col(c) for c in gold_cols]
    p_processed = [process_col(c) for c in pred_cols]

    matches = 0
    p_matched_indices = set()
    for g_col in g_processed:
        for i, p_col in enumerate(p_processed):
            if i in p_matched_indices:
                continue
            
            # Check for exact content match of the column
            if g_col == p_col:
                matches += 1
                p_matched_indices.add(i)
                break

    return matches / len(gold_cols) if gold_cols else 0.0
    

def compute_scores(pred_sql: str, gold_sql: str, db_id: str) -> tuple:
    """
    Calculates:
    1. Subset match: 1.0 if Gold result is a subset of Pred result
    2. Overlap Score (Jaccard): Intersection / Union of rows
    3. 3. Column Match (Uses the original lists to preserve column structure)
    """
    gold_status, gold_rows, pred_status, pred_rows = _execute_both_queries(gold_sql, pred_sql, db_id)
    
    if not (pred_status and gold_status):
        return (0.0, 0.0, 0.0)

    # 2. Convert to sets for row-level overlap metrics
    pred_set = set(pred_rows)
    gold_set = set(gold_rows)

    # Subset Match (Gold subset of Pred)
    subset_score = 1.0 if gold_set.issubset(pred_set) else 0.0
    
    # Row Overlap (Jaccard)
    intersection = len(gold_set & pred_set)
    union = len(gold_set | pred_set)
    row_overlap_score = intersection / union if union > 0 else 0.0
    
    # 3. Column Match (Uses the original lists to preserve column structure)
    col_match_score = column_level_match(pred_rows, gold_rows)

    return (subset_score, row_overlap_score, col_match_score)


def evaluate():
    print(f"Loading full fine-tuned model from: {CHECKPOINT_PATH}")
    tokenizer =  AutoTokenizer.from_pretrained(CHECKPOINT_PATH)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Could not load adapter: {e}")
        return

    print("Loading Dataset...")
    dataset = load_dataset("json", data_files={'test': TEST_DATA_PATH}, split="test")

    metrics = {"subset": [], "row_overlap": [], "col_match": []}

    print("Starting Evaluation...")
    for item in tqdm(dataset):
        messages = item['prompt']
        gold_sql = item['query']
        db_id = item['db_id']

        text_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        pred_sql = extract_query_from_response(generated_text)

        sub, row_over, col_match = compute_scores(pred_sql, gold_sql, db_id)
        
        metrics["subset"].append(sub)
        metrics["row_overlap"].append(row_over)
        metrics["col_match"].append(col_match)

        print("\n" + "="*40)
        print(f"Evaluation Results on {len(dataset)} examples:")
        print(f"Subset Match (Gold in Pred):   {sum(metrics['subset'])/len(metrics['subset']):.4f}")
        print(f"Row Overlap (Row Jaccard):     {sum(metrics['row_overlap'])/len(metrics['row_overlap']):.4f}")
        print(f"Column Match (Avg Col Recall): {sum(metrics['col_match'])/len(metrics['col_match']):.4f}")
        print("="*40)




if __name__ == "__main__":
    evaluate()