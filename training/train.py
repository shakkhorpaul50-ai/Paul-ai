import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "HuggingFaceTB/SmolLM-135M"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "training_data.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "smolLM-chat-reasoning-lora")


def load_training_data() -> Dataset:
    data = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_chat(example, tokenizer):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    tokenized = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading training data...")
    dataset = load_training_data()
    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    print("Saving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Done. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
