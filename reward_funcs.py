import re
import os
import sqlite3
import sqlglot
from pathlib import Path
from difflib import SequenceMatcher
import sqlparse


DATABASE_BASE_DIRECTORY = "/dataset/spider_data/database"


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


            
        


    
        
        


        