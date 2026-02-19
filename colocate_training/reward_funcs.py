import re
import sqlite3
from collections import Counter
from pathlib import Path

def extract_query_from_response(text: str) -> str:
    m = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    return text.strip()

def query_match_reward_func(completions, query, **kwargs):
    rewards = []
    for complete, q in zip(completions, query):
        extracted_query = extract_query_from_response(complete[0]['content'])
        rewards.append(1.0 if extracted_query.strip() == q.strip() else 0.0)

    return rewards

def execution_match_reward_func(completions, query_result, db_id, **kwargs):
    rewards = []
    for completion, gold_rows, db in zip(completions, query_result, db_id):
        try:
            sql = extract_query_from_response(completion[0]["content"])
            db_path = Path(f"./spider_data/test_database/{db}/{db}.sqlite")
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            pred_rows = [str(r) for r in cur.execute(sql).fetchall()]
            con.close()
            rewards.append(1.0 if Counter(pred_rows) == Counter(gold_rows) else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards