"""
Fine‑tune a transformer for AI vs Human classification.

This script trains a Hugging Face transformer model end‑to‑end on your
labelled data using the Trainer API. It automatically uses the
combined dataset splits (`train_full_clean.csv`, `val_full_clean.csv`)
if they exist, otherwise it falls back to the original Kaggle splits.

Prerequisites: install the `datasets` and `transformers` libraries and
ensure you have enough compute (fine‑tuning even a small model like
distilroberta-base or bert-base-uncased may require a GPU for faster
training).

Usage::

    python src/finetune_transformer.py --model_name roberta-base --epochs 3

The script writes the fine‑tuned model and tokenizer to the `models/`
directory along with a JSON file of evaluation metrics.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


def load_splits() -> Dict[str, pd.DataFrame]:
    """Load train/val/test DataFrames, preferring combined splits if present."""
    base = Path("data/processed")
    splits = {}
    # Combined dataset
    if (base / "train_full_clean.csv").exists():
        splits["train"] = pd.read_csv(base / "train_full_clean.csv")
        splits["val"] = pd.read_csv(base / "val_full_clean.csv")
        if (base / "test_full_clean.csv").exists():
            splits["test"] = pd.read_csv(base / "test_full_clean.csv")
        print("[data] Using combined dataset for fine‑tuning.")
    else:
        # Fall back to Kaggle
        splits["train"] = pd.read_csv(base / "train_clean.csv")
        splits["val"] = pd.read_csv(base / "val_clean.csv")
        if (base / "test_clean.csv").exists():
            splits["test"] = pd.read_csv(base / "test_clean.csv")
        print("[data] Using Kaggle dataset for fine‑tuning.")
    return splits


def prepare_datasets(splits: Dict[str, pd.DataFrame], tokenizer, max_len: int) -> Dict[str, Dataset]:
    """Convert Pandas DataFrames into Hugging Face Datasets and tokenize."""
    def tokenize_function(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
        )

    hf_datasets = {}
    for split, df in splits.items():
        # Map labels to integers: AI -> 1, Human -> 0
        labels = df["label"].map(lambda x: 1 if str(x).strip().lower() == "ai" else 0).tolist()
        dataset = Dataset.from_dict({"text": df["text"].astype(str).tolist(), "label": labels})
        dataset = dataset.map(tokenize_function, batched=True)
        hf_datasets[split] = dataset
    return hf_datasets


def compute_metrics(pred):
    """Custom metric function for the Trainer."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main(args: argparse.Namespace) -> None:
    splits = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    datasets = prepare_datasets(splits, tokenizer, args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    output_dir = Path("models") / f"finetuned_{args.model_name.replace('/', '_')}"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    # Save model and tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    # Save metrics
    metrics_path = output_dir / "finetune_metrics.json"
    with open(metrics_path, "w") as f:
        import json
        json.dump(eval_metrics, f, indent=4)
    print("[model] Fine‑tuned model saved to", output_dir)
    print("[metrics] Saved evaluation metrics to", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine‑tune a transformer model for AI vs Human classification.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Hugging Face model name (e.g. 'distilroberta-base', 'roberta-base')")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--epochs", type=float, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient")
    args = parser.parse_args()
    main(args)