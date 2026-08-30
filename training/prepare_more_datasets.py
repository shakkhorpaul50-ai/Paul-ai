import csv
import json
import random
import shutil
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PARTS_DIR = DATA_DIR / "processed" / "parts"
FINAL = DATA_DIR / "processed" / "training_data.jsonl"

SYSTEM_BN_ASSISTANT = "You are a helpful Bangla AI assistant. Be concise and clear."
SYSTEM_BN_FRIEND = "You are a friendly Bangla companion. Keep replies natural and conversational."
SYSTEM_BN_TUTOR = "You are a Bangla math tutor. Solve problems step by step in Bangla."
SYSTEM_BN_WRITER = "You are a fluent writer in Bangla. Continue writing naturally."
SYSTEM_EN_WRITER = "You are a fluent English writer. Continue writing naturally."
SYSTEM_EN_ASSISTANT = "You are a helpful AI assistant."

rng = random.Random(7)


def write(items, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  -> {out} ({len(items)} rows)")


def part_bangla_chat():
    print("[1/5] Bangla multi-turn chat (DailyDialogue bn)")
    tar_path = hf_hub_download(
        "csebuetnlp/dailydialogue_bn", "data/dailydialogue_bn.tar.bz2",
        repo_type="dataset", local_dir=str(RAW_DIR),
    )
    exdir = RAW_DIR / "bangla_nlg_dailydialogue"
    if not (exdir / "train.jsonl").exists():
        shutil.rmtree(exdir, ignore_errors=True)
        exdir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:bz2") as tf:
            tf.extractall(exdir)

    items = []
    seen = set()
    for split in ("train", "test", "validation"):
        src = exdir / f"{split}.jsonl"
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                try:
                    turns = json.loads(line.strip())["source"]
                except Exception:
                    continue
                turns = [t.strip() for t in turns if t and t.strip()]
                if len(turns) < 2:
                    continue
                msgs = [{"role": "system", "content": SYSTEM_BN_FRIEND}]
                for i, t in enumerate(turns[:12]):
                    msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
                key = json.dumps(msgs, ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key)
                items.append({"messages": msgs})
    write(items, PARTS_DIR / "bn_chat.jsonl")


def part_bangla_reasoning():
    print("[2/5] Bangla reasoning (z4hid/bengali-math-cot, sampled 1/4)")
    csv_path = hf_hub_download(
        "z4hid/bengali-math-cot", "bangla-math-cot-dataset.csv",
        repo_type="dataset", local_dir=str(RAW_DIR),
    )
    items = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 4 != 0:
                continue
            problem = (row.get("problem") or "").strip()
            solution = (row.get("solution") or "").strip()
            if not problem or not solution:
                continue
            items.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_BN_TUTOR},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": solution},
                ]
            })
    write(items, PARTS_DIR / "bn_reasoning.jsonl")


def part_english_instruction():
    print("[3/5] English instruction (yahma/alpaca-cleaned)")
    json_path = hf_hub_download(
        "yahma/alpaca-cleaned", "alpaca_data_cleaned.json",
        repo_type="dataset", local_dir=str(RAW_DIR),
    )
    with open(json_path, encoding="utf-8") as f:
        rows = json.load(f)
    items = []
    for r in rows:
        instruction = str(r.get("instruction") or "").strip()
        user_input = str(r.get("input") or "").strip()
        output = str(r.get("output") or "").strip()
        user_content = instruction + (f"\n\n{user_input}" if user_input else "")
        if not user_content or not output:
            continue
        items.append({
            "messages": [
                {"role": "system", "content": SYSTEM_EN_ASSISTANT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        })
    write(items, PARTS_DIR / "en_instruction.jsonl")


def _gen_from_texts(texts, system, prompt_fmt, user_fmt, limit, prefix_len=280, cont_len=420):
    items = []
    seen = set()
    for text in texts:
        text = text.strip()
        if len(text) < prefix_len + 60:
            continue
        prefix = text[:prefix_len]
        continuation = text[prefix_len:prefix_len + cont_len]
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_fmt.format(prefix=prefix)},
            {"role": "assistant", "content": continuation.strip()},
        ]
        key = user_fmt.format(prefix=prefix)
        if key in seen:
            continue
        seen.add(key)
        items.append({"messages": msgs})
        if len(items) >= limit:
            break
    return items


def part_generative():
    print("[4/5] Generative Bangla (Wikipedia bn)")
    bn_parts = [
        "20231101.bn/train-00000-of-00002.parquet",
        "20231101.bn/train-00001-of-00002.parquet",
    ]
    texts = []
    for fn in bn_parts:
        p = hf_hub_download("wikimedia/wikipedia", fn, repo_type="dataset", local_dir=str(RAW_DIR))
        t = pq.read_table(str(p), columns=["text"])
        texts.extend(t.column("text").to_pylist())
    bn_items = _gen_from_texts(
        texts, SYSTEM_BN_WRITER,
        "Continue writing in Bangla:\n{prefix}",
        "{prefix}",
        limit=60000,
    )
    write(bn_items, PARTS_DIR / "gen_bn.jsonl")

    print("[5/5] Generative English (TinyStories train shard 0)")
    en_path = hf_hub_download(
        "roneneldan/TinyStories", "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
        repo_type="dataset", local_dir=str(RAW_DIR),
    )
    t = pq.read_table(str(en_path), columns=["text"])
    en_texts = t.column("text").to_pylist()
    en_items = _gen_from_texts(
        en_texts, SYSTEM_EN_WRITER,
        "Continue the story in English:\n{prefix}",
        "{prefix}",
        limit=60000,
    )
    write(en_items, PARTS_DIR / "gen_en.jsonl")


def merge():
    print("Merging into final training_data.jsonl ...")
    parts = sorted(PARTS_DIR.glob("*.jsonl")) if PARTS_DIR.exists() else []
    srcs = [FINAL] + parts
    tmp = FINAL.with_name("training_data.merged.jsonl")
    total = 0
    with open(tmp, "w", encoding="utf-8") as out:
        for src in srcs:
            with open(src, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total += 1
    tmp.replace(FINAL)
    shutil.rmtree(PARTS_DIR, ignore_errors=True)
    print(f"TOTAL: {total} examples -> {FINAL} ({FINAL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    part_bangla_chat()
    part_bangla_reasoning()
    part_english_instruction()
    part_generative()
    merge()