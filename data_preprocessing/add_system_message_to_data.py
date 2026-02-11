from datasets import load_dataset
import re


SYSTEM_MESSAGE = """You are an expert SQL generator.
Given a database schema and a question, you must: 
1. Reason about the query logic inside <think> tags.
2. Only output the valid SQLite query inside <sql> tags.
"""

"""Default dataset has column named "question" for x value, which doesn't match with Huggingface's TRL library's expected "prompt" column."""
DATASET = load_dataset("spider", split="train")


def format_prompt(data_point: dict) -> dict:
    user_question = f'Question: {data_point["question"]}'

    messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_question}]
    
    return {"prompt": messages}


def extract_query_from_sql_tags(text: str) -> str:
    m = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL /re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    return text.strip()


if __name__ == "__main__":
    processed_dataset = DATASET.map(format_prompt)
    for key, value in processed_dataset[0].items():
        print(f"{key}: {value}")
