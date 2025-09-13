# scripts/finetune_transformer_fast.py
from __future__ import annotations

import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

SEED = 42
MAX_LEN = int(os.getenv("TRANSFORMER_MAX_LEN", "256"))
BASE_MODEL = os.getenv("BASE_MODEL", "distilroberta-base")
OUT_DIR = Path("models/finetuned_distilroberta-base")
DATA_DIR = Path("data/processed")
# Preference order: richer merged file if it exists, else original
DEFAULT_DATASETS = [
    DATA_DIR / "train_full.csv",
    DATA_DIR / "dataset.csv",
]

LABEL2ID = {"Human": 0, "AI": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


@dataclass
class SimpleDataset:
    encodings: Dict[str, torch.Tensor]
    labels: List[int]
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)


def load_dataframe() -> pd.DataFrame:
    for p in DEFAULT_DATASETS:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No dataset found. Expected one of: " + ", ".join(map(str, DEFAULT_DATASETS)))

    # Normalise columns
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    # Clean
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len().between(30, 1500)]
    df = df[df["label"].isin(LABEL2ID.keys())].copy()
    df.dropna(subset=["text", "label"], inplace=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def tokenize_batch(tokenizer, texts: List[str]):
    # Pad to MAX_LEN so each batch is a proper tensor
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataframe()
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"].tolist(),
        [LABEL2ID[l] for l in df["label"].tolist()],
        test_size=0.12,
        random_state=SEED,
        stratify=df["label"].values,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    m = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_enc = tokenize_batch(tokenizer, X_train)
    val_enc   = tokenize_batch(tokenizer, X_val)

    # convert lists to tensors
    # train_enc = {k: torch.tensor(v) for k, v in train_enc.items()}
    # val_enc   = {k: torch.tensor(v) for k, v in val_enc.items()}

    train_ds = SimpleDataset(train_enc, y_train)
    val_ds   = SimpleDataset(val_enc, y_val)

    args = TrainingArguments(
        output_dir=str(OUT_DIR / "trainer"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=SEED,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=m,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save final model + tokenizer
    m.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    # Save label mapping
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    print(f"[ok] Saved finetuned model to: {OUT_DIR}")


if __name__ == "__main__":
    main()
