import re
import sqlite3
import sqlglot
import sqlparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache


DATABASE_BASE_DIRECTORY = Path("dataset/spider_data/database")
MAX_WORKERS = 16  # Adjust based on CPU cores (matches your cpus-per-task)


def get_completion_text(complete) -> str:
    """Handle both string completions and message-dict completions."""
    if isinstance(complete, str):
        return complete
    return complete[0]['content']


def extract_query_from_response(text: str) -> str:
    """Extracts SQL from <sql> tags or returns the raw text."""
    m = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


@lru_cache(maxsize=None)
def _get_db_path(db_id: str) -> Path:
    """Resolves and caches the database path for a given DB ID."""
    base = DATABASE_BASE_DIRECTORY / db_id
    for ext in (".sqlite", ".db"):
        p = base / f"{db_id}{ext}"
        if p.exists():
            return p
    return Path()


def _safe_execute_sql(db_path, query):
    """
    Executes SQL safely in a separate thread.
    Creates a fresh connection per call to ensure thread safety.
    Returns (rows, cols) or (None, None) on failure.
    """
    try:
        with closing(sqlite3.connect(db_path, timeout=10)) as conn:
            cur = conn.cursor()
            cur.execute(query)
            rows = [str(r) for r in cur.fetchall()]
            cols = {desc[0].lower() for desc in cur.description} if cur.description else set()
            return rows, cols
    except Exception:
        return None, None


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
        extracted = extract_query_from_response(get_completion_text(complete))
        db_path = _get_db_path(db)
        _, _, = _safe_execute_sql(db_path, extracted)
        success = _ is not None
        rewards.append(1.0 if success else 0.0)
    return rewards


def query_ngram_comparison_reward(completions, query_toks, **kwargs):
    """
    Reward: SequenceMatcher ratio between predicted and gold tokens.
    """
    rewards = []
    for complete, gold_tok in zip(completions, query_toks):
        try:
            extracted = extract_query_from_response(get_completion_text(complete))
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
            extracted = extract_query_from_response(get_completion_text(complete))
            pred_items = extract_schema_items(extracted)
            gold_items = extract_schema_items(gold_sql)
            rewards.append(jaccard_similarity(pred_items, gold_items))
        except Exception:
            rewards.append(0.0)
    return rewards


def comprehensive_execution_reward_func(completions, query_result, query_result_columns, db_id, **kwargs):
    """Parallelized execute pred_query and calculate row similarity and column similarity scores"""
    def process_single(args):
        complete, gold_rows, gold_column_names, db = args
        try:
            pred_query = extract_query_from_response(get_completion_text(complete))
            db_path = _get_db_path(db)

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
            tp = sum(intersection.values())
            fp = sum((pred_counter - gold_counter).values())
            fn = sum((gold_counter - pred_counter).values())

            if tp == 0:
                row_score = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                row_score = 2 * (precision * recall) / (precision + recall)

            return (0.3 * col_score) + (0.7 * row_score)

        except Exception:
            return 0.0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(process_single, zip(completions, query_result, query_result_columns, db_id)))


def subset_match_reward_func(completions, query_result, db_id, **kwargs):
    """Parallelized execute pred_query and give partial credit based on row overlap"""
    def process_single(args):
        complete, gold_rows, db = args
        try:
            pred_query = extract_query_from_response(get_completion_text(complete))
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
            elif pred_set.issubset(gold_set):
                return len(pred_set) / len(gold_set)
            elif gold_set.issubset(pred_set):
                return len(gold_set) / len(pred_set) if pred_set else 0.0
            else:
                inter = len(pred_set.intersection(gold_set))
                union = len(pred_set.union(gold_set))
                return inter / union if union > 0 else 0.0

        except Exception:
            return 0.0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(process_single, zip(completions, query_result, db_id)))


def execution_exact_match_reward_func(completions, query_result, db_id, **kwargs):
    """Parallelized execute pred_query and give 1.0 for exact match of gold_rows and pred_rows"""
    def process_single(args):
        complete, gold_rows, db = args
        try:
            pred_query = extract_query_from_response(get_completion_text(complete))
            db_path = _get_db_path(db)

            pred_rows, _ = _safe_execute_sql(db_path, pred_query)
            if pred_rows is None:
                return 0.0

            return 1.0 if Counter(pred_rows) == Counter(gold_rows) else 0.0

        except Exception:
            return 0.0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(process_single, zip(completions, query_result, db_id)))