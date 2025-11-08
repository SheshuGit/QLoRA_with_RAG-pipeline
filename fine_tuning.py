"""
Fine-tuning script for FLAN-T5 (seq2seq) models with optional QLoRA.
Now supports both training and validation datasets.

Requirements:
    - data/rag_train.json
    - data/rag_val.json
Both containing:
    {"question": "...", "context": "...", "answer": "..."}

The script:
    ✅ Loads both datasets
    ✅ Prepares seq2seq tokens
    ✅ Fine-tunes with validation monitoring
    ✅ Saves model to models/finetuned/
"""

import os
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from config import Config

# Optional: QLoRA imports
try:
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False


# -----------------------------------------------------------
# 1️⃣ Load and Prepare Datasets
# -----------------------------------------------------------
def load_json_dataset(path: str) -> Dataset:
    """Load question-context-answer JSON file and return as HuggingFace Dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for item in data:
        q, c, a = item.get("question", "").strip(), item.get("context", "").strip(), item.get("answer", "").strip()
        if not q or not a:
            continue
        inp = f"Question: {q}\nContext: {c}" if c else f"Question: {q}"
        records.append({"input_text": inp, "target_text": a})

    print(f"✅ Loaded {len(records)} samples from {path}")
    return Dataset.from_list(records)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_input_len=512, max_target_len=128):
    """Tokenize dataset for seq2seq model."""
    def preprocess(example):
        model_inputs = tokenizer(
            example["input_text"],
            max_length=max_input_len,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            example["target_text"],
            max_length=max_target_len,
            truncation=True,
            padding="max_length"
        )["input_ids"]
        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        labels = [(t if t != pad_token_id else -100) for t in labels]
        model_inputs["labels"] = labels
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])


# -----------------------------------------------------------
# 2️⃣ Load Model (with optional QLoRA)
# -----------------------------------------------------------
def setup_model(model_name: str, use_qlora: bool, device: str):
    """Load model with QLoRA if available, else standard seq2seq."""
    if use_qlora and BNB_AVAILABLE and device.startswith("cuda"):
        print("🧠 Loading with QLoRA (4-bit quantization)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("✅ QLoRA setup complete.")
        return model
    else:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        print("🧩 Loading model in standard mode...")
        return AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")


# -----------------------------------------------------------
# 3️⃣ Fine-tuning Pipeline
# -----------------------------------------------------------
def fine_tune():
    config = Config()
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Load datasets
    train_path = os.path.join(config.DATA_DIR, "rag_train.json")
    val_path = os.path.join(config.DATA_DIR, "rag_val.json")

    train_ds = load_json_dataset(train_path)
    val_ds = load_json_dataset(val_path)

    tokenized_train = tokenize_dataset(train_ds, tokenizer)
    tokenized_val = tokenize_dataset(val_ds, tokenizer)

    # Load model
    model = setup_model(config.MODEL_NAME, getattr(config, "USE_Q_LORA", True), device)

    # Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", return_tensors="pt")

    # Training arguments
    output_dir = os.path.join(config.MODELS_DIR, "finetuned")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=getattr(config, "EPOCHS", 3),
        per_device_train_batch_size=getattr(config, "BATCH_SIZE", 4),
        gradient_accumulation_steps=getattr(config, "GRAD_ACC", 1),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=getattr(config, "LEARNING_RATE", 2e-4),
        logging_steps=20,
        load_best_model_at_end=True,
        report_to="none",
        fp16=device == "cuda",
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("🚀 Starting fine-tuning with validation monitoring...")
    trainer.train()
    print("✅ Fine-tuning complete.")

    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    fine_tune()
