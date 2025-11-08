# fine_tuning.py
"""
Fine-tuning script for FLAN-T5 style seq2seq models.
- Provides a safe QLoRA path when CUDA + bitsandbytes are available.
- Falls back to regular float32 fine-tune on CPU (or no bnb).
- Ensures tokenizer produces input_ids/attention_mask and labels for seq2seq training.
"""

import os
import json
from pathlib import Path
from typing import Dict
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

# Optional imports for QLoRA
try:
    from transformers import BitsAndBytesConfig
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False


def load_training_data(data_dir: str) -> Dataset:
    """
    Expects data_dir/rag_train.json with a list of objects:
    {"question": "...", "context": "...", "answer": "..."}
    Returns a HuggingFace Dataset with columns: input_text, target_text
    """
    path = os.path.join(data_dir, "rag_train.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"rag_train.json not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        q = item.get("question", "").strip()
        c = item.get("context", "").strip()
        a = item.get("answer", "").strip()
        if not q or not a:
            continue
        # Input is question + context (context can be empty)
        if c:
            inp = f"Question: {q}\nContext: {c}"
        else:
            inp = f"Question: {q}"
        rows.append({"input_text": inp, "target_text": a})

    print(f"Loaded {len(rows)} training samples from {path}")
    return Dataset.from_list(rows)


def prepare_tokenized_datasets(dataset: Dataset, tokenizer: AutoTokenizer, max_input_length=512, max_target_length=128):
    """Tokenize and return tokenized dataset with labels prepared for seq2seq."""
    def _tokenize(batch: Dict):
        inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )
        targets = tokenizer(
            batch["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding="max_length"
        )

        # For seq2seq models we set labels to target input_ids, and replace pad token id by -100
        labels = targets["input_ids"]
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        labels = [[(t if t != pad_token_id else -100) for t in label] for label in labels]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["input_text", "target_text"])
    return tokenized


def setup_model_and_peft(model_name: str, use_bnb: bool, device: str):
    """
    Load seq2seq model. If use_bnb is True and environment supports it, load with BitsAndBytes (QLoRA).
    Otherwise, load normal float32 model.
    Returns (model, using_peft_flag)
    """
    if use_bnb and BNB_AVAILABLE and device.startswith("cuda"):
        print("Attempting to load model with QLoRA (4-bit) + LoRA adapters...")
        # bitsandbytes quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # LoRA config for seq2seq
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_2_SEQ_LM",
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_cfg)
        print("✅ QLoRA + LoRA adapters loaded.")
        model.print_trainable_parameters()
        return model, True
    else:
        if use_bnb and not BNB_AVAILABLE:
            print("⚠ bitsandbytes/peft not available — falling back to standard model load.")
        if device.startswith("cuda"):
            print("Loading model in float16 on GPU (no QLoRA).")
            dtype = torch.float16
        else:
            print("Loading model in float32 (CPU or no GPU).")
            dtype = torch.float32

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device.startswith("cuda") else None
        )
        return model, False


def fine_tune():
    config = Config()

    # Create dirs
    Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    # T5 needs pad token set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Load dataset
    raw_ds = load_training_data(config.DATA_DIR)

    # Prepare dataset for seq2seq
    tokenized_ds = prepare_tokenized_datasets(raw_ds, tokenizer,
                                              max_input_length=config.MAX_INPUT_LEN if hasattr(config, "MAX_INPUT_LEN") else 512,
                                              max_target_length=config.MAX_TARGET_LEN if hasattr(config, "MAX_TARGET_LEN") else 128)

    # Load model (QLoRA if possible and requested)
    want_bnb = getattr(config, "USE_Q_LORA", True)
    model, used_peft = setup_model_and_peft(config.MODEL_NAME, use_bnb=want_bnb, device=device)

    # Data collator (handles dynamic padding and labels)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest", return_tensors="pt")

    # Training arguments - adjust for your machine
    output_dir = os.path.join(config.MODELS_DIR, "finetuned")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size= config.BATCH_SIZE if hasattr(config, "BATCH_SIZE") else (8 if device == "cuda" else 4),
        gradient_accumulation_steps= getattr(config, "GRAD_ACC", 1),
        num_train_epochs= getattr(config, "EPOCHS", 3),
        logging_steps=50,
        save_strategy="no",
        learning_rate= getattr(config, "LEARNING_RATE", 2e-4),
        fp16=(device=="cuda"),
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator
    )

    print("🚀 Starting training. This may take some time...")
    trainer.train()
    print("✅ Training finished. Saving model...")

    # Save model + tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    fine_tune()
