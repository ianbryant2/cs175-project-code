import re
import sqlite3
import sqlglot
import sqlparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
from contextlib import closing
from functools import lru_cache

# ---------------- CONFIGURATION ----------------

TRAIN_DATABASE_BASE_DIRECTORY = Path("dataset/spider_data/database")
TEST_DATABASE_BASE_DIRECTORY = Path("dataset/spider_data/test_database")

# ---------------- HELPER FUNCTIONS ----------------

def sanitize_sql(sql: str) -> str:
    return sql.encode("ascii", errors="ignore").decode("ascii")

def get_completion_text(complete) -> str:
    """Handle both string completions and message-dict completions."""
    if isinstance(complete, str):
        return complete
    if isinstance(complete, list) and len(complete) > 0:
        return complete[0].get('content', '')
    if isinstance(complete, dict):
        return complete.get('content', '')
    return str(complete)


def extract_query_from_response(text: str) -> str:
    """Extracts SQL from <sql> tags or returns the raw text."""
    m = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL | re.IGNORECASE)
    return sanitize_sql(m.group(1).strip() if m else text.strip())


@lru_cache(maxsize=None)
def _get_db_path(base_directory: Path, db_id: str) -> Path:
    """Resolves and caches the database path for a given DB ID."""
    base = base_directory / db_id
    for ext in (".sqlite", ".db"):
        p = base / f"{db_id}{ext}"
        if p.exists():
            return p
    return Path()


def _interrupt_handler():
    """Callback to interrupt long-running SQL queries."""
    return 1


def _safe_execute_sql(db_path, query):
    """
    Executes SQL safely.
    1. Opens in Read-Only mode (prevents file locking deadlocks on HPC).
    2. Sets a progress handler to interrupt infinite loops.
    """
    if not db_path:
        return None, None
    try:
        # uri=True allows us to pass mode=ro (Read Only)
        # timeout=5 prevents waiting for locks indefinitely
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)) as conn:
            
            # Set a progress handler to abort queries that take too many opcodes
            conn.set_progress_handler(_interrupt_handler, 100000) 
            
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


# ---------------- CORE REWARD FUNCTIONS ----------------

def syntax_check_reward(completions, db_id, **kwargs):
    """Reward: 1.0 if the SQL executes without error, 0.0 otherwise."""
    rewards = []
    for complete, db in zip(completions, db_id):
        extracted = extract_query_from_response(get_completion_text(complete))
        db_path = _get_db_path(TRAIN_DATABASE_BASE_DIRECTORY, db)
        if not db_path.exists():
            db_path = _get_db_path(TEST_DATABASE_BASE_DIRECTORY, db)
            
        rows, _ = _safe_execute_sql(db_path, extracted)
        success = rows is not None
        rewards.append(1.0 if success else 0.0)
    return rewards


def query_ngram_comparison_reward(completions, query_toks, **kwargs):
    """Reward: SequenceMatcher ratio between predicted and gold tokens."""
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
    """Reward: Jaccard similarity of tables/columns used in prediction vs gold."""
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
    """
    Sequential execution reward.
    Score = 0.3 * (Column Intersection) + 0.7 * (Row Content F1 Score)
    """
    rewards = []
    for complete, gold_rows, gold_column_names, db in zip(completions, query_result, query_result_columns, db_id):
        try:
            pred_query = extract_query_from_response(get_completion_text(complete))
            db_path = _get_db_path(TRAIN_DATABASE_BASE_DIRECTORY, db)
            if db_path == Path():
                db_path = _get_db_path(TEST_DATABASE_BASE_DIRECTORY, db)

            pred_rows, pred_cols = _safe_execute_sql(db_path, pred_query)
            if pred_rows is None:
                rewards.append(0.0)
                continue

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
            
            intersection_counts = pred_counter & gold_counter
            tp = sum(intersection_counts.values())
            fp = sum((pred_counter - gold_counter).values())
            fn = sum((gold_counter - pred_counter).values())

            if tp == 0:
                row_score = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                row_score = 2 * (precision * recall) / (precision + recall)

            rewards.append((0.3 * col_score) + (0.7 * row_score))

        except Exception:
            rewards.append(0.0)
            
    return rewards


def subset_match_reward_func(completions, query_result, db_id, **kwargs):
    """Sequential execute pred_query and give partial credit based on row overlap"""
    rewards = []
    for complete, gold_rows, db in zip(completions, query_result, db_id):
        try:
            pred_query = extract_query_from_response(get_completion_text(complete))
            db_path = _get_db_path(TEST_DATABASE_BASE_DIRECTORY, db)

            pred_rows, _ = _safe_execute_sql(db_path, pred_query)
            if pred_rows is None:
                rewards.append(0.0)
                continue

            pred_set = set(pred_rows)
            gold_set = set(gold_rows)

            if not gold_set:
                rewards.append(1.0 if not pred_set else 0.0)
            elif pred_set == gold_set:
                rewards.append(1.0)
            elif pred_set.issubset(gold_set):
                rewards.append(len(pred_set) / len(gold_set))
            elif gold_set.issubset(pred_set):
                rewards.append(len(gold_set) / len(pred_set) if pred_set else 0.0)
            else:
                inter = len(pred_set.intersection(gold_set))
                union = len(pred_set.union(gold_set))
                rewards.append(inter / union if union > 0 else 0.0)

        except Exception:
            rewards.append(0.0)
    return rewards


def execution_exact_match_reward_func(completions, query_result, db_id, **kwargs):
    """Sequential execute pred_query and give 1.0 for exact match of gold_rows and pred_rows"""
    rewards = []
    for complete, gold_rows, db in zip(completions, query_result, db_id):
        try:
            pred_query = extract_query_from_response(get_completion_text(complete))
            db_path = _get_db_path(TEST_DATABASE_BASE_DIRECTORY, db)

            pred_rows, _ = _safe_execute_sql(db_path, pred_query)
            if pred_rows is None:
                rewards.append(0.0)
                continue

            rewards.append(1.0 if Counter(pred_rows) == Counter(gold_rows) else 0.0)

        except Exception:
            rewards.append(0.0)
    return rewards


# ---------------- SCHEDULING LOGIC CLASS ----------------

class ScheduledReward:
    """
    Callable class that manages reward scheduling by holding a reference 
    to the trainer to access current training steps.
    """
    def __init__(self):
        self.trainer = None
        # FIX: The trainer needs a __name__ attribute to log the function name
        self.__name__ = "scheduled_sql_reward"

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _get_progress(self) -> float:
        if self.trainer is None or self.trainer.state is None:
            return 0.0
        
        # Handle HuggingFace TrainerState
        max_steps = self.trainer.state.max_steps
        global_step = self.trainer.state.global_step
        
        if max_steps > 0:
            return min(max(global_step / max_steps, 0.0), 1.0)
        return 0.0

    def _coerce_rewards(self, component, n):
        """Ensures reward components are length-n float lists."""
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

    def __call__(self, completions, **kwargs):
        """
        The main entry point called by GRPOTrainer.
        """
        n = len(completions)
        progress = self._get_progress()
        
        # Schedule: Early (<10%), Mid (<40%), Late (>40%)
        if progress < 0.10:
            weights = {"schema": 0.30, "ngram": 0.25, "syntax": 0.35, "execution": 0.10}
        elif progress < 0.40:
            weights = {"schema": 0.20, "ngram": 0.20, "syntax": 0.20, "execution": 0.40}
        else:
            weights = {"schema": 0.10, "ngram": 0.10, "syntax": 0.10, "execution": 0.70}

        # Extract columns from kwargs
        query = kwargs.get("query")
        query_toks = kwargs.get("query_toks")
        query_result = kwargs.get("query_result")
        query_result_columns = kwargs.get("query_result_columns")
        db_id = kwargs.get("db_id")

        # Calculate Components
        schema_component = schema_linking_reward(completions, query=query) if query else None
        ngram_component = query_ngram_comparison_reward(completions, query_toks=query_toks) if query_toks else None
        syntax_component = syntax_check_reward(completions, db_id=db_id) if db_id else None
        
        execution_component = None
        if query_result is not None and query_result_columns is not None and db_id is not None:
            execution_component = comprehensive_execution_reward_func(
                completions,
                query_result=query_result,
                query_result_columns=query_result_columns,
                db_id=db_id,
            )

        # Coerce to valid lists
        schema_rewards = self._coerce_rewards(schema_component, n)
        ngram_rewards = self._coerce_rewards(ngram_component, n)
        syntax_rewards = self._coerce_rewards(syntax_component, n)
        execution_rewards = self._coerce_rewards(execution_component, n)

        # Weighted Combination
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