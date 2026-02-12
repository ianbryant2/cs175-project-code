import re

def extract_query_from_response(text: str) -> str:
    m = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    
    return text.strip()

def query_match_reward_func(completions, query, **kwargs):
    rewards = []
    for complete, q in zip(completions, query):
        extracted_query = extract_query_from_response(complete[0]['content'])
        rewards.append(1.0 if extracted_query.strip() == q.strip() else 0.0)

    return rewards