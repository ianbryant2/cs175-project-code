from datasets import load_dataset
import re


SYSTEM_MESSAGE = """"You are a SQL expert. Output only a valid and executable SQL in <sql> tags."""
DATASET = load_dataset("spider", split="train")


def format_prompt(data_point: dict) -> dict:
    user_question = f'Question: {data_point["question"]}'

    messages = [{"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_question}]
    
    return {"prompt": messages}


def extract_query_from_response(text: str) -> str:
    m = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL /re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    return text.strip()


def extract_thinking_from_response(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL /re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    return text.strip()


if __name__ == "__main__":
    processed_dataset = DATASET.map(format_prompt)
    for i in range(10):
        print(f"Prompt {i}: {processed_dataset[i]['question']}")
