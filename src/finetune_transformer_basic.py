"""
Fine‑tune a transformer classifier without relying on the `datasets` library.

This script is an alternative to `finetune_transformer.py` for environments
where installing the `datasets` package is problematic. It uses pure
PyTorch DataLoader objects and a simple training loop. A GPU is highly
recommended for reasonable training times.

Usage::

    python src/finetune_transformer_basic.py --model_name bert-base-uncased --epochs 3

The script saves the fine‑tuned model and tokenizer under `models/finetuned_BASIC_<model_name>` and prints evaluation metrics.
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW  # Use PyTorch's AdamW to avoid import issues with transformers


class TextDataset(Dataset):
    """A simple PyTorch dataset for text classification."""
    def __init__(self, texts, labels, tokenizer, max_len: int = 256):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[dict, int]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path("data/processed")
    if (base / "train_full_clean.csv").exists():
        train_df = pd.read_csv(base / "train_full_clean.csv")
        val_df = pd.read_csv(base / "val_full_clean.csv")
        print("[data] Using combined dataset for basic fine‑tuning.")
    else:
        train_df = pd.read_csv(base / "train_clean.csv")
        val_df = pd.read_csv(base / "val_clean.csv")
        print("[data] Using Kaggle dataset for basic fine‑tuning.")
    return train_df, val_df


def prepare_dataloaders(train_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer, batch_size: int, max_len: int):
    def df_to_dataset(df):
        labels = df["label"].map(lambda x: 1 if str(x).strip().lower() == "ai" else 0).tolist()
        return TextDataset(df["text"].astype(str).tolist(), labels, tokenizer, max_len)
    train_dataset = df_to_dataset(train_df)
    val_dataset = df_to_dataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs: int):
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{int(epochs)} - train loss: {avg_loss:.4f}")
        evaluate(model, val_loader, device)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**batch_inputs)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    print(f"  val accuracy: {acc:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Basic transformer fine‑tuner without datasets library.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name or path")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_len", type=int, default=256, help="Max token length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_df, val_df = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = prepare_dataloaders(train_df, val_df, tokenizer, args.batch_size, args.max_len)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)
    total_steps = len(train_loader) * int(args.epochs)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    train(model, train_loader, val_loader, optimizer, scheduler, device, args.epochs)
    # Final evaluation
    metrics = evaluate(model, val_loader, device)
    # Save model
    out_dir = Path("models") / f"finetuned_BASIC_{args.model_name.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    import json
    with open(out_dir / "finetune_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Saved fine‑tuned model to", out_dir)


if __name__ == "__main__":
    main()
