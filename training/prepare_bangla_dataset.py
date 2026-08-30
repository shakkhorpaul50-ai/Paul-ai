import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

REPO = "BanglaLLM/bangla-alpaca-orca"
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT = DATA_DIR / "processed" / "training_data.jsonl"
DEFAULT_SYSTEM = "You are a helpful Bangla AI assistant. Be concise and clear."


def download_parquet() -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    repo_files = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet"]
    paths = []
    for rf in repo_files:
        local = hf_hub_download(repo_id=REPO, filename=rf, repo_type="dataset", local_dir=str(RAW_DIR))
        paths.append(Path(local))
        print(f"  downloaded {local}")
    return paths


def iter_rows(paths: list[Path]):
    for p in paths:
        table = pq.read_table(str(p))
        for batch in table.to_batches():
            for i in range(batch.num_rows):
                row = {col: batch.column(col)[i].as_py() for col in batch.column_names}
                yield row


def build_messages(row: dict) -> dict | None:
    instruction = str(row.get("instruction") or "").strip()
    user_input = str(row.get("input") or "").strip()
    output = str(row.get("output") or "").strip()
    system_prompt = str(row.get("system_prompt") or "").strip()

    user_content = instruction
    if user_input:
        user_content = f"{instruction}\n\n{user_input}"

    if not user_content or not output:
        return None

    return {
        "messages": [
            {"role": "system", "content": system_prompt if system_prompt else DEFAULT_SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Download BanglaLLM/bangla-alpaca-orca and convert to messages JSONL")
    parser.add_argument("--max-examples", type=int, default=0, help="Cap output rows (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Downloading {REPO} ...")
    parquet_paths = download_parquet()

    print("Converting to messages format ...")
    seen = set()
    items = []
    total = 0
    for row in iter_rows(parquet_paths):
        total += 1
        item = build_messages(row)
        if item is None:
            continue
        key = hashlib.sha256(json.dumps(item, ensure_ascii=False).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
        if args.max_examples and len(items) >= args.max_examples:
            break

    print(f"  scanned {total} rows, kept {len(items)} unique")

    rng = random.Random(args.seed)
    rng.shuffle(items)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Saved {len(items)} examples to {OUTPUT} ({size_mb:.1f} MB)")

    print("\nSample:")
    for item in items[:3]:
        for m in item["messages"]:
            content = m["content"].replace("\n", " ")
            print(f"  [{m['role']}] {content[:120]}")
        print("  ---")


if __name__ == "__main__":
    main()