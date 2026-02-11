import re

def extract_query_from_response(text: str) -> str:
    m = re.search(r"<sql>(.*?)</sql>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    return text.strip()

def query_match_reward_func(completions, query, **kwargs):
    return [1.0 if extract_query_from_response(response).strip() == query.strip() else 0.0 for response in completions]