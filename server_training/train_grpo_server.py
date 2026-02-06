# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from trl.rewards import accuracy_reward
import re
import os

def extract_final(text: str) -> str:
    # try common "final answer" patterns; fall back to last token-ish chunk
    m = re.search(r"(?:final answer|answer)\s*[:\-]\s*(.*)$", text, re.I | re.M)
    if m:
        return m.group(1).strip()
    return text.strip().splitlines()[-1].strip()

def simple_accuracy_reward(completions, solution, **kwargs):
    preds = [extract_final(c[0]["content"]) for c in completions]
    golds = solution
    out = []
    for p, g in zip(preds, golds):
        out.append(1.0 if p.strip() == g.strip() else 0.0)
    return out

print("Start loading")
dataset = load_dataset("trl-lib/DeepMath-103K", split="train")

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode='server',
    vllm_server_base_url = os.environ['VLLM_BASE_URL']
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=simple_accuracy_reward,
    train_dataset=dataset,
    args=training_args
)
print("Start training")
trainer.train()
