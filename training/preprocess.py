import csv
import json
import os
import glob

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")


def find_file(pattern: str) -> str | None:
    matches = glob.glob(os.path.join(DATA_DIR, "**", pattern), recursive=True)
    return matches[0] if matches else None


def load_chat_data() -> list[dict]:
    path = find_file("*chatbot_conversations*")
    if not path:
        print("Chat dataset not found. Run download_datasets.py first.")
        return []

    conversations = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row.get("conversation_id", "")
            if cid not in conversations:
                conversations[cid] = []
            conversations[cid].append(row)

    out = []
    for cid, turns in conversations.items():
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        for turn in turns:
            role = "user" if turn.get("role", "").strip() == "user" else "assistant"
            messages.append({"role": role, "content": turn.get("message", "")})
        if len(messages) >= 3:
            out.append({"messages": messages})
    return out


def load_reasoning_data() -> list[dict]:
    path = find_file("*Reasoning*") or find_file("*reasoning*")
    if not path:
        print("Reasoning dataset not found.")
        return []

    out = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", row.get("question", ""))
            answer = row.get("answer", row.get("reasoning", ""))
            if prompt and answer:
                out.append({
                    "messages": [
                        {"role": "system", "content": "You are a helpful problem-solving assistant. Think step by step."},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                })
    return out


def load_math_data() -> list[dict]:
    path = find_file("*math*") or find_file("*Math*")
    if not path:
        print("Math dataset not found.")
        return []

    out = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", row.get("Problem", ""))
            a = row.get("answer", row.get("Solution", ""))
            if q and a:
                out.append({
                    "messages": [
                        {"role": "system", "content": "You are a math tutor. Solve problems step by step."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ]
                })
    return out


def merge_and_save():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    all_data.extend(load_chat_data())
    all_data.extend(load_reasoning_data())
    all_data.extend(load_math_data())

    print(f"Total training examples: {len(all_data)}")

    out_path = os.path.join(OUTPUT_DIR, "training_data.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    merge_and_save()
