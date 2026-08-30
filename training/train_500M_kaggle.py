"""
Qwen2.5-0.5B-Instruct LoRA fine-tune for Kaggle T4 x2 within 1 hour
Dataset: Kaggle Input My-ai-1 -> G:\AI\online\qwen500M-bangla-lora.zip -> 20k capped
Trained: 1500 steps (~1.2 epochs), FP16 LoRA r=32, 355MB Q3_K_M GGUF fits Render Free 512MB
Usage:
  Kaggle Notebook: Add Input -> My-ai-1, Accelerator T4 x2, Internet ON, Run all cells
  Local merge: python training/quantize.py --qtype q3_k_m
"""
import glob
import hashlib
import json
import os
import random
import shutil
import zipfile

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_EXAMPLES = 20000
OUTPUT_DIR = "/kaggle/working/models/qwen500M-bangla-lora"
# Fallback for local run
LOCAL_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "models", "qwen500M-bangla-lora")


def find_and_prepare_data():
    # Kaggle Input is nested: /kaggle/input/datasets/shakkhorpaul/my-ai-1/...
    # Generic recursive copy handles any dataset name
    kaggle_input = "/kaggle/input"
    working_data = "/kaggle/working/data"
    if os.path.exists(kaggle_input):
        os.makedirs(working_data, exist_ok=True)
        # Unzip any zips
        for zp in glob.glob(os.path.join(kaggle_input, "**/*.zip"), recursive=True):
            print(f"Unzipping {zp}")
            with zipfile.ZipFile(zp) as z:
                z.extractall(working_data)
        # Copy all files recursively
        for src in glob.glob(os.path.join(kaggle_input, "**/*"), recursive=True):
            if os.path.isfile(src) and not src.endswith(".zip"):
                rel = os.path.relpath(src, kaggle_input)
                dst = os.path.join(working_data, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        data_root = working_data
    else:
        data_root = os.path.join(os.path.dirname(__file__), "..", "data")
        if not os.path.exists(working_data):
            data_root = os.path.join(os.path.dirname(__file__), "..", "data")

    candidates = glob.glob(os.path.join(data_root, "**/training_data.jsonl"), recursive=True)
    if not candidates:
        # also check G:\AI\online\data fallback
        candidates = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "**/*.jsonl"), recursive=True)
    assert candidates, f"training_data.jsonl not found under {data_root}"
    src = candidates[0]
    print(f"Loading {src} ({os.path.getsize(src)/1e9:.2f} GB)")
    return src


def load_items(src):
    items, seen = [], set()
    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "messages" not in obj:
                continue
            k = hashlib.sha256(json.dumps(obj, ensure_ascii=False).encode()).hexdigest()
            if k in seen:
                continue
            seen.add(k)
            items.append(obj)
            if len(items) >= MAX_EXAMPLES:
                break
    random.Random(42).shuffle(items)
    print(f"Kept {len(items)} examples (capped {MAX_EXAMPLES})")
    return items[:MAX_EXAMPLES]


def main():
    src = find_and_prepare_data()
    items = load_items(src)
    assert len(items) > 0, "No data found"

    is_kaggle = os.path.exists("/kaggle/working")
    output_dir = OUTPUT_DIR if is_kaggle else LOCAL_OUTPUT

    ds = Dataset.from_list(items)
    split = ds.train_test_split(test_size=0.01, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train {len(train_ds)} Eval {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def fmt(ex):
        txt = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
        t = tokenizer(txt, truncation=True, max_length=512, padding="max_length")
        t["labels"] = t["input_ids"].copy()
        return t

    train_ds = train_ds.map(fmt, remove_columns=train_ds.column_names, desc="tok train")
    eval_ds = eval_ds.map(fmt, remove_columns=eval_ds.column_names, desc="tok eval")

    # FP16 LoRA - avoids bitsandbytes/triton mismatch on Kaggle T4 x2
    # Qwen2.5-0.5B 24 layers, 896 hidden, requires 256-alignment fallback for Q3_K_M
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Fix for deprecated evaluation_strategy and to ensure grads
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        max_steps=1500,  # 1500 * 16 batch = 24k samples ~1.2 epochs of 19800
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        fp16=True,
        optim="adamw_torch",
        gradient_checkpointing=False,  # False fits T4 16GB for 0.5B; True needs enable_input_require_grads + use_reentrant=False
        report_to="none",
        dataloader_num_workers=2,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    print("Training start...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    # Optional: zip for Kaggle Output
    if is_kaggle:
        import subprocess

        subprocess.run(["zip", "-r", "/kaggle/working/qwen500M-bangla-lora.zip", output_dir], check=False)
        print("Zipped to /kaggle/working/qwen500M-bangla-lora.zip")

        # Merge for test
        from peft import PeftModel

        print("Merging...")
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
        peft_model = PeftModel.from_pretrained(base, output_dir)
        merged = peft_model.merge_and_unload()
        merged.save_pretrained("/kaggle/working/models/merged")
        tokenizer.save_pretrained("/kaggle/working/models/merged")
        print("Merged to /kaggle/working/models/merged")

        # Quick Bangla test
        inputs = tokenizer("হ্যালো, তুমি কে? বাংলায় বলো।", return_tensors="pt").to(merged.device)
        with torch.no_grad():
            out = merged.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True)
        print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
