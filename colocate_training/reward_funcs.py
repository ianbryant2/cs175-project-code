import re
import os
import sqlite3
import sqlglot
from difflib import SequenceMatcher
import sqlparse


DATABASE_BASE_DIRECTORY = "../dataset/spider_data/database"


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


def connect_to_spider_db(db_name: str):
    db_path = os.path.join(DATABASE_BASE_DIRECTORY, db_name, f"{db_name}.sqlite")
    if not os.path.exists(db_path):
        print(f"FileNotFound: {db_path}")

    return sqlite3.connect(db_path)

    
def query_execution_reward(completions, db_id, **kwargs):
    rewards = []
    for complete, db_name in zip(completions, db_id):
        extracted_query = extract_query_from_response(complete[0]['content'])

        if not extracted_query:
            rewards.append(-1.0)
            continue
            
        try:
            cursor = connect_to_spider_db(db_name).cursor()
            cursor.execute(extracted_query)
            results = cursor.fetchall()
            if results:
                rewards.append(1.0)

            else:
                rewards.append(0.5)
                
        except sqlite3.OperationalError as e:
            rewards.append(-0.5)

        except sqlite3.Error as e:
            rewards.append(-0.5)

    return rewards


def query_ngram_comparison_reward(completions, query_toks, **kwargs):
    rewards = []
    for complete, query_tok in zip(completions, query_toks):
        try:
            extracted_generated_query = extract_query_from_response(complete[0]['content'])
            generated_tok = [str(token) for token in sqlparse(extracted_generated_query)[0].flatten()\
                         if not token.is_whitespace]
    
            similarity = SequenceMatcher(None, query_tok, generated_tok).ratio()
            rewards.append(similarity)
        except Exception as e:
            rewards.append(0.0)

    return rewards




    
        
        


        