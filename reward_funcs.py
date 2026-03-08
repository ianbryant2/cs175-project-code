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


def _training_progress(trainer_state) -> float:
    """
    Returns normalized training progress in [0, 1].
    """
    if trainer_state is None:
        return 0.0

    max_steps = getattr(trainer_state, "max_steps", None)
    global_step = getattr(trainer_state, "global_step", 0)

    if isinstance(max_steps, int) and max_steps > 0:
        return min(max(global_step / max_steps, 0.0), 1.0)

    return 0.0


def _scheduled_weights(progress: float):
    """
    Three-phase reward schedule:
    - Early: emphasize syntax/schema/token similarity.
    - Mid: shift weight toward execution.
    - Late: prioritize execution correctness.
    """
    if progress < 0.10:
        return {
            "schema": 0.30,
            "ngram": 0.25,
            "syntax": 0.35,
            "execution": 0.10,
        }
    if progress < 0.40:
        return {
            "schema": 0.20,
            "ngram": 0.20,
            "syntax": 0.20,
            "execution": 0.40,
        }
    return {
        "schema": 0.10,
        "ngram": 0.10,
        "syntax": 0.10,
        "execution": 0.70,
    }


def _coerce_rewards(component, n):
    """
    Ensures reward components are length-n float lists.
    """
    if component is None:
        return [0.0] * n

    values = []
    for i in range(n):
        if i < len(component) and component[i] is not None:
            try:
                values.append(float(component[i]))
            except Exception:
                values.append(0.0)
        else:
            values.append(0.0)
    return values


def scheduled_sql_reward(completions, trainer_state=None, **kwargs):
    """
    Meta-reward wrapper that schedules weights across training.
    """
    n = len(completions)
    progress = _training_progress(trainer_state)
    weights = _scheduled_weights(progress)

    query = kwargs.get("query")
    query_toks = kwargs.get("query_toks")
    query_result = kwargs.get("query_result")
    query_result_columns = kwargs.get("query_result_columns")
    db_id = kwargs.get("db_id")

    schema_component = None
    if query is not None:
        schema_component = schema_linking_reward(completions, query=query)

    ngram_component = None
    if query_toks is not None:
        ngram_component = query_ngram_comparison_reward(completions, query_toks=query_toks)

    syntax_component = None
    if db_id is not None:
        syntax_component = syntax_check_reward(completions, db_id=db_id)

    execution_component = None
    if query_result is not None and query_result_columns is not None and db_id is not None:
        execution_component = comprehensive_execution_reward_func(
            completions,
            query_result=query_result,
            query_result_columns=query_result_columns,
            db_id=db_id,
        )

    schema_rewards = _coerce_rewards(schema_component, n)
    ngram_rewards = _coerce_rewards(ngram_component, n)
    syntax_rewards = _coerce_rewards(syntax_component, n)
    execution_rewards = _coerce_rewards(execution_component, n)

    combined = []
    for i in range(n):
        reward = (
            weights["schema"] * schema_rewards[i]
            + weights["ngram"] * ngram_rewards[i]
            + weights["syntax"] * syntax_rewards[i]
            + weights["execution"] * execution_rewards[i]
        )
        combined.append(reward)

    return combined
    
