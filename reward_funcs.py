import re
import sqlite3
import sqlglot
import sqlparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
from contextlib import closing


DATABASE_BASE_DIRECTORY = Path("dataset/spider_data/database")


def _get_db_path(db_id: str) -> Path:
    """Resolves the database path for a given DB ID."""
    base = DATABASE_BASE_DIRECTORY / db_id
    for ext in (".sqlite", ".db"):
        p = base / f"{db_id}{ext}"
        if p.exists():
            return p
    return Path()


def extract_query_from_response(text: str) -> str:
    """Extracts SQL from <sql> tags or returns the raw text."""
    m = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def safe_execute_sql(db_id: str, query: str):
    """
    Executes SQL safely and returns rows, columns, and a success flag.
    """
    db_path = _get_db_path(db_id)
    if not db_path.exists():
        return [], set(), False

    try:
        with closing(sqlite3.connect(db_path, timeout=10)) as conn:
            cur = conn.cursor()
            cur.execute(query)
            # Convert rows to strings for consistent comparison
            rows = [str(r) for r in cur.fetchall()]
            cols = {desc[0].lower() for desc in cur.description} if cur.description else set()
            return rows, cols, True
    except Exception:
        return [], set(), False


def extract_schema_items(sql: str, dialect="sqlite") -> set:
    """Extracts table and column names using sqlglot."""
    items = set()
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
        if tree:
            for node in tree.find_all(sqlglot.exp.Table, sqlglot.exp.Column):
                if node.name:
                    items.add(node.name.lower())
    except Exception:
        pass
    return items


def jaccard_similarity(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0

# ---------------- REWARD FUNCTIONS ----------------

def syntax_check_reward(completions, db_id, **kwargs):
    """
    Reward: 1.0 if the SQL executes without error, 0.0 otherwise.
    """
    rewards = []
    for complete, db in zip(completions, db_id):
        extracted = extract_query_from_response(complete[0]['content'])
        _, _, success = safe_execute_sql(db, extracted)
        rewards.append(1.0 if success else 0.0)
    return rewards


def query_ngram_comparison_reward(completions, query_toks, **kwargs):
    """
    Reward: SequenceMatcher ratio between predicted and gold tokens.
    """
    rewards = []
    for complete, gold_tok in zip(completions, query_toks):
        try:
            extracted = extract_query_from_response(complete[0]['content'])
            # Parse and flatten tokens, ignoring whitespace
            pred_tok = [str(t) for t in sqlparse.parse(extracted)[0].flatten() if not t.is_whitespace]
            rewards.append(SequenceMatcher(None, gold_tok, pred_tok).ratio())
        except Exception:
            rewards.append(0.0)
    return rewards


def schema_linking_reward(completions, query, **kwargs):
    """
    Reward: Jaccard similarity of tables/columns used in prediction vs gold.
    """
    rewards = []
    for complete, gold_sql in zip(completions, query):
        try:
            extracted = extract_query_from_response(complete[0]['content'])
            pred_items = extract_schema_items(extracted)
            gold_items = extract_schema_items(gold_sql)
            rewards.append(jaccard_similarity(pred_items, gold_items))
        except Exception:
            rewards.append(0.0)
    return rewards


def comprehensive_execution_reward_func(completions, query_result, query_result_columns, db_id, **kwargs):
    """
    Reward: Weighted score based on column match (structure) and row F1 score (content).
    """
    rewards = []
    iterator = zip(completions, query_result, query_result_columns, db_id)
    
    for complete, gold_rows, gold_cols_list, db in iterator:
        pred_query = extract_query_from_response(complete[0]['content'])
        pred_rows, pred_cols, success = safe_execute_sql(db, pred_query)

        if not success:
            rewards.append(0.0)
            continue

        # Score 1: Column Overlap
        gold_cols = set(gold_cols_list) if gold_cols_list else set()
        col_score = jaccard_similarity(pred_cols, gold_cols)

        # Score 2: Row Content F1 (using Counters for multisets)
        pred_counter = Counter(pred_rows)
        gold_counter = Counter(gold_rows)
        
        tp = sum((pred_counter & gold_counter).values())
        fp = sum((pred_counter - gold_counter).values())
        fn = sum((gold_counter - pred_counter).values())

        if tp == 0:
            row_score = 0.0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            row_score = 2 * (precision * recall) / (precision + recall)

        # Final Weighted Score (30% Structure, 70% Content)
        rewards.append((0.3 * col_score) + (0.7 * row_score))

    return rewards
    