import csv
import glob
import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED = DATA_DIR / "processed"
FINAL = PROCESSED / "training_data.jsonl"
TMP_CHAT = PROCESSED / "tmp_chat_en.jsonl"
TMP_REASON = PROCESSED / "tmp_reasoning_en.jsonl"


def find_file(pattern: str) -> Path | None:
    matches = glob.glob(str(DATA_DIR / "**" / pattern), recursive=True)
    return Path(matches[0]) if matches else None


def write_line(f, item):
    f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_chat(path: Path, out_path: Path) -> int:
    conversations = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        total = 0
        for row in reader:
            total += 1
            cid = row.get("conversation_id", "")
            if cid not in conversations:
                conversations[cid] = []
            role = "user" if (row.get("role", "") or "").strip() == "user" else "assistant"
            content = row.get("message", "") or ""
            if content.strip():
                conversations[cid].append((role, content))
            if total % 500_000 == 0:
                print(f"    chat: scanned {total//1000}k rows, {len(conversations)} convos")

    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for cid, turns in conversations.items():
            msgs = [{"role": "system", "content": "You are a helpful AI assistant."}]
            msgs.extend({"role": r, "content": c} for r, c in turns)
            if len(msgs) >= 3:
                write_line(out, {"messages": msgs})
                count += 1
    return count


def convert_reasoning(path: Path, out_path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = (row.get("prompt") or row.get("question") or "").strip()
            answer = (row.get("answer") or row.get("reasoning") or "").strip()
            if prompt and answer:
                write_line(out, {
                    "messages": [
                        {"role": "system", "content": "You are a helpful problem-solving assistant. Think step by step."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                })
                count += 1
    return count


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    chat_path = find_file("*chatbot_conversations*.csv")
    if chat_path is None or chat_path.name.endswith("_bn.csv"):
        chat_path = find_file("chatbot_conversations.csv")
    print(f"English chat source: {chat_path}")
    n_chat = convert_chat(chat_path, TMP_CHAT) if chat_path else 0
    print(f"English chat conversations: {n_chat}")

    reason_path = find_file("*Reasoning*.csv") or find_file("*reasoning*.csv")
    if reason_path and reason_path.name.endswith("_bn.csv"):
        reason_path = find_file("*Reasoning*.csv")
    print(f"English reasoning source: {reason_path}")
    n_reason = convert_reasoning(reason_path, TMP_REASON) if reason_path else 0
    print(f"English reasoning examples: {n_reason}")

    bangla_path = PROCESSED / "training_data.jsonl"
    n_bn = sum(1 for _ in open(bangla_path, encoding="utf-8"))
    print(f"Bangla examples: {n_bn}")

    source_files = []
    if TMP_CHAT.exists():
        source_files.append(TMP_CHAT)
    if TMP_REASON.exists():
        source_files.append(TMP_REASON)

    tmp_final = PROCESSED / "training_data.combined.jsonl"
    total = 0
    with open(tmp_final, "w", encoding="utf-8") as out:
        for src in source_files:
            with open(src, encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        out.write(line)
                        total += 1
        with open(bangla_path, encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    out.write(line)
                    total += 1

    tmp_final.replace(FINAL)
    TMP_CHAT.unlink(missing_ok=True)
    TMP_REASON.unlink(missing_ok=True)

    size_mb = FINAL.stat().st_size / 1e6
    print(f"\nMerged TOTAL: {total} examples -> {FINAL} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()