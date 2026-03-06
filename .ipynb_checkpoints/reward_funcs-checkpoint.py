import re
import os
import sqlite3
import sqlglot
from pathlib import Path
from difflib import SequenceMatcher
import sqlparse
from collections import Counter
from contextlib import closing


DATABASE_BASE_DIRECTORY = "dataset/spider_data/database"
GOLD_CACHE = {}


def extract_query_from_response(text: str) -> str:
    m = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    return text.strip()


def query_match_reward_func(completions, query, **kwargs):
    rewards = []
    for complete, q in zip(completions, query):
        extracted_query = extract_query_from_response(complete[0]['content'])
        rewards.append(1.0 if extracted_query.strip() == q.strip() else 0.0)

    return rewards


"""Execute the generated query against the corresponding table and 
give a score of 0/1, return a list of rewards"""
def syntax_check_reward(completions, db_id, **kwargs):
    rewards = []
    for complete, database in zip(completions, db_id):
        try:
            extracted = extract_query_from_response(complete[0]['content'])
            db_path = Path(os.path.join(DATABASE_BASE_DIRECTORY, database, f"{database}.sqlite"))
            if not db_path.exists():
                db_path = Path(os.path.join(DATABASE_BASE_DIRECTORY, database, f"{database}.db"))
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(extracted)
            conn.close()
            rewards.append(1.0)
            
        except sqlite3.Error:
            rewards.append(0.0)
            
        except Exception:
            rewards.append(0.0)

    return rewards


"""Perform an N-Gram comparison between the LLM generated query and a 
gold query and return a list of reward scores (0.0-1.0)"""
def query_ngram_comparison_reward(completions, query_toks, **kwargs):
    rewards = []
    for complete, query_tok in zip(completions, query_toks):
        try:
            extracted_generated_query = extract_query_from_response(complete[0]['content'])
            generated_tok = [str(token) for token in sqlparse.parse(extracted_generated_query)[0].flatten()\
                         if not token.is_whitespace]
    
            similarity = SequenceMatcher(None, query_tok, generated_tok).ratio()
            rewards.append(similarity)
        except Exception as e:
            rewards.append(0.0)

    return rewards


def extract_schema_items(sql: str, dialect="sqlite") -> set:
    items = set()
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
        if tree is None:
            return items
        for table in tree.find_all(sqlglot.exp.Table):
            if table.name:
                items.add(table.name.lower())
        for col in tree.find_all(sqlglot.exp.Column):
            if col.name:
                items.add(col.name.lower())
    except Exception:
        pass
    return items


def jaccard_similarity(set_a: set, set_b: set) -> float:
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return len(intersection) / len(union)


"""Extract all the schema items from the generated query and the gold
query and calculate the jaccard similarity, return a list of rewards"""
def schema_linking_reward(completions, query, **kwargs):
    rewards = []
    for complete, q in zip(completions, query):
        if not complete:
            rewards.append(0.0)
            continue
        try:
            extracted = extract_query_from_response(complete[0]['content'])
            pred_schema_items = extract_schema_items(extracted)
            query_schema_items = extract_schema_items(q)
            rewards.append(jaccard_similarity(pred_schema_items, query_schema_items))

        except:
            rewards.append(0.0)

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


def _get_db_path(db_id: str) -> Path:
    base = Path(DATABASE_BASE_DIRECTORY) / db_id
    for ext in (".sqlite", ".db"):
        p = base / f"{db_id}{ext}"
        if p.exists():
            return p
    return Path()


"""Execute gold_query and pred_query and calculate row similarity and column similarity scores"""
def comprehensive_execution_reward_func(completions, query, db_id, **kwargs):
    rewards = []
    for complete, gold_query, db in zip(completions, query, db_id):
        try:
            pred_query = extract_query_from_response(complete[0]['content'])
            db_path = _get_db_path(db)
            with closing(sqlite3.connect(db_path)) as conn:
                cur = conn.cursor()
    
                #Execute Generated Query
                cur.execute(pred_query)
                pred_rows = [str(r) for r in cur.fetchall()]
                pred_cols = set([desc[0].lower() for desc in cur.description])
    
                # Execute Gold Query (to get ground truth structure)
                cache_key = (db, gold_query)
                if cache_key in GOLD_CACHE:
                    gold_rows, gold_cols = GOLD_CACHE[cache_key]
                else:
                    cur.execute(gold_query)
                    gold_rows = [str(r) for r in cur.fetchall()]
                    gold_cols = set([desc[0].lower() for desc in cur.description])
                    GOLD_CACHE[cache_key] = (gold_rows, gold_cols)

            # Score 1: Structural Accuracy (Column)
            if not gold_cols:
                col_score = 0.0
            else:
                intersection = pred_cols.intersection(gold_cols)
                union = pred_cols.union(gold_cols)
                col_score = len(intersection) / len(union)
            
            # Score 2: Data Accuracy (Row)
            pred_counter = Counter(pred_rows)
            gold_counter = Counter(gold_rows)
                         
            # Intersection counts common elements including duplicates
            # e.g., Gold={A:2}, Pred={A:1} -> Intersection={A:1}
            intersection = pred_counter & gold_counter 
            tp = sum(intersection.values()) # True Positives
            
            fp = sum((pred_counter - gold_counter).values()) # Extra rows (False Positives)
            fn = sum((gold_counter - pred_counter).values()) # Missing rows (False Negatives)

            if tp == 0:
                row_score = 0.0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                row_score = 2 * (precision * recall) / (precision + recall)

            # 4. Final Weighted Score
            # You can adjust weights. 0.3 for columns (structure) and 0.7 for rows (content) is a good mix.
            final_score = (0.3 * col_score) + (0.7 * row_score)
            rewards.append(final_score)

        except Exception:
            rewards.append(0.0)

    return rewards


            
        


    
        
        


        