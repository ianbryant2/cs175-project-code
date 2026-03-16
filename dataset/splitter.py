import argparse
import json
from pathlib import Path


DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parent
    / "spider_data"
    / "preprocessed"
    / "preprocessed_test_spider.json"
)


def split_preprocessed_train_by_query_toks(num_partitions: int,input_path: Path = DEFAULT_INPUT_PATH) -> list[Path]:
    """
    Split preprocessed_train_spider.json into N partitions ordered by query_toks length.
    Partition 1 contains shortest SQL token sequences, partition N contains longest.
    """
    if num_partitions <= 0:
        raise ValueError("num_partitions must be >= 1")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected input JSON to be a list of datapoints.")

    sorted_data = sorted(data, key=lambda x: len(x.get("query_toks", [])))

    total = len(sorted_data)
    base_size = total // num_partitions
    remainder = total % num_partitions

    output_dir = input_path.parent / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    start = 0
    for i in range(num_partitions):
        part_size = base_size + (1 if i < remainder else 0)
        end = start + part_size
        split_chunk = sorted_data[start:end]

        output_path = output_dir / f"{i + 1}.json"
        with output_path.open("w") as f:
            json.dump(split_chunk, f, indent=4)

        written_files.append(output_path)
        print(f"Wrote {output_path} ({len(split_chunk)} records)")
        start = end

    return written_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_partitions",
        type=int,
        help="Number of partitions to split preprocessed_train_spider.json into.",
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to file containing preprocessed data"
    )
    args = parser.parse_args()
    split_preprocessed_train_by_query_toks(args.num_partitions, input_path=Path(args.input_path))
