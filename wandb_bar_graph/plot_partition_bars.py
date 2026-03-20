#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRIC_KEY = "eval/rewards/execution_exact_match_reward_func/mean"
MODEL_DIRS = ("deepseek", "qwen2_5", "qwen3")
BAR_LABELS = [
    "Shortest Queries\n(1/5)",
    "Short-Medium\nQueries (2/5)",
    "Medium Queries\n(3/5)",
    "Medium-Long\nQueries (4/5)",
    "Longest Queries\n(5/5)",
]
BAR_COLORS = ["#62B6DB", "#5DCC83", "#FF8A1A", "#9B3FA9", "#F44747"]


def parse_partition_index(file_name: str) -> int:
    digits = re.findall(r"\d+", file_name)
    if not digits:
        raise ValueError(f"No numeric partition index found in filename: {file_name}")
    return int(digits[-1])


def read_model_partition_scores(model_dir: Path) -> dict[int, float]:
    scores = {}
    for json_file in sorted(model_dir.glob("*.json")):
        partition = parse_partition_index(json_file.name)
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if METRIC_KEY not in data:
            raise KeyError(f"Missing metric key '{METRIC_KEY}' in {json_file}")
        scores[partition] = float(data[METRIC_KEY]) * 100.0
    return scores


def plot_single_model(model_name: str, scores: dict[int, float], out_path: Path) -> None:
    partitions = [1, 2, 3, 4, 5]
    missing = [p for p in partitions if p not in scores]
    if missing:
        raise ValueError(f"{model_name}: missing partition files for split(s): {missing}")

    y = [scores[p] for p in partitions]
    x = np.arange(len(partitions))

    plt.figure(figsize=(12, 7), dpi=150)
    bars = plt.bar(x, y, color=BAR_COLORS, width=0.6, edgecolor="black", linewidth=0.5)

    plt.title(f"Query Accuracy vs. Query Length ({model_name})", fontsize=22, pad=16)
    plt.ylabel("Percentage Accuracy (%)", fontsize=16)
    plt.xlabel("Query Length Buckets", fontsize=16, labelpad=8)
    plt.xticks(x, BAR_LABELS, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="-", alpha=0.3)
    plt.gca().set_axisbelow(True)

    for bar, val in zip(bars, y):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one query-length bar chart per model folder from W&B summary JSON files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder containing deepseek/, qwen2_5/, qwen3/ (default: wandb_bar_graph/).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Output directory for PNG files.",
    )
    args = parser.parse_args()

    for model_dir_name in MODEL_DIRS:
        model_dir = args.root / model_dir_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model folder not found: {model_dir}")
        scores = read_model_partition_scores(model_dir)
        out_path = args.out_dir / f"{model_dir_name}_execution_exact_match_bar.png"
        plot_single_model(model_dir_name, scores, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
