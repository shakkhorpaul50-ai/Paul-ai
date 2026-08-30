import csv
import json
import os
import time
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_model(device="cuda"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def translate_batch(texts, model, tokenizer, device="cuda", max_length=512):
    if not texts:
        return []
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with __import__("torch").no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("ben_Beng"),
            max_length=max_length,
            num_beams=4,
        )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def save_checkpoint(data, output_path, mode="w"):
    with open(output_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def translate_reasoning(input_path, output_path, model, tokenizer, device, chunk_size=100):
    rows = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    print(f"Reasoning dataset: {total} rows")
    translated = []

    for i in range(0, total, chunk_size):
        chunk = rows[i : i + chunk_size]
        prompts = [r.get("prompt", "") for r in chunk]
        answers = [r.get("answer", "") for r in chunk]
        reasoning = [r.get("reasoning", "") for r in chunk]

        t_prompts = translate_batch(prompts, model, tokenizer, device)
        t_answers = translate_batch(answers, model, tokenizer, device)
        t_reasoning = translate_batch(reasoning, model, tokenizer, device)

        for j, r in enumerate(chunk):
            translated.append({
                "prompt": t_prompts[j],
                "reasoning": t_reasoning[j],
                "answer": t_answers[j],
            })

        print(f"  Translated {min(i + chunk_size, total)}/{total}")
        save_checkpoint(translated, output_path)

    return translated


def translate_chat(input_path, output_path, model, tokenizer, device, chunk_size=1000):
    rows = []
    with open(input_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    print(f"Chat dataset: {total} rows")
    translated = []

    for i in range(0, total, chunk_size):
        chunk = rows[i : i + chunk_size]
        messages = [r.get("message", "") for r in chunk]

        t_messages = translate_batch(messages, model, tokenizer, device)

        for j, r in enumerate(chunk):
            translated.append({
                "conversation_id": r.get("conversation_id", ""),
                "turn": r.get("turn", ""),
                "role": r.get("role", ""),
                "intent": r.get("intent", ""),
                "message": t_messages[j],
            })

        print(f"  Translated {min(i + chunk_size, total)}/{total}")
        save_checkpoint(translated, output_path)

    return translated


def translate_benchmark(input_path, output_path, model, tokenizer, device, chunk_size=100):
    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))

    total = len(rows)
    print(f"Benchmark dataset: {total} rows")
    translated = []

    for i in range(0, total, chunk_size):
        chunk = rows[i : i + chunk_size]
        questions = [r.get("question", "") for r in chunk]

        t_questions = translate_batch(questions, model, tokenizer, device)

        for j, r in enumerate(chunk):
            translated.append({
                "id": r.get("id"),
                "category": r.get("category", ""),
                "question": t_questions[j],
                "answer": r.get("answer"),
                "difficulty": r.get("difficulty"),
                "type": r.get("type", ""),
            })

        print(f"  Translated {min(i + chunk_size, total)}/{total}")
        save_checkpoint(translated, output_path)

    return translated


def main():
    parser = argparse.ArgumentParser(description="Translate datasets to Bangla using NLLB-200")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--chunk-size", type=int, default=100, help="Batch size for translation")
    parser.add_argument("--reasoning-only", action="store_true", help="Translate only reasoning dataset")
    parser.add_argument("--chat-only", action="store_true", help="Translate only chat dataset")
    parser.add_argument("--benchmark-only", action="store_true", help="Translate only benchmark dataset")
    args = parser.parse_args()

    model, tokenizer = load_model(args.device)

    reasoning_path = DATA_DIR / "1B_Model_Low_Reasoning_Data.csv"
    reasoning_out = DATA_DIR / "1B_Model_Low_Reasoning_Data_bn.csv"
    chat_path = DATA_DIR / "chatbot_conversations.csv"
    chat_out = DATA_DIR / "chatbot_conversations_bn.csv"
    benchmark_path = DATA_DIR / "benchmark_llm_reasoning.jsonl"
    benchmark_out = DATA_DIR / "benchmark_llm_reasoning_bn.jsonl"

    do_all = not (args.reasoning_only or args.chat_only or args.benchmark_only)

    if do_all or args.reasoning_only:
        if reasoning_path.exists():
            print("\n--- Translating Reasoning Dataset ---")
            translate_reasoning(reasoning_path, reasoning_out, model, tokenizer, args.device, args.chunk_size)
        else:
            print(f"Skipping: {reasoning_path} not found")

    if do_all or args.chat_only:
        if chat_path.exists():
            print("\n--- Translating Chat Dataset ---")
            translate_chat(chat_path, chat_out, model, tokenizer, args.device, args.chunk_size)
        else:
            print(f"Skipping: {chat_path} not found")

    if do_all or args.benchmark_only:
        if benchmark_path.exists():
            print("\n--- Translating Benchmark Dataset ---")
            translate_benchmark(benchmark_path, benchmark_out, model, tokenizer, args.device, args.chunk_size)
        else:
            print(f"Skipping: {benchmark_path} not found")

    print("\nTranslation complete!")


if __name__ == "__main__":
    main()
