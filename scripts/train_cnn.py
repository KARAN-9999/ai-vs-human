# scripts/train_cnn_char.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = Path("data/processed")
OUT_DIR  = Path("models/cnn_small")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 96 printable ASCII chars (plus a few extras)
CHARS = (
    "\n\r\t " +
    "abcdefghijklmnopqrstuvwxyz" +
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
    "0123456789" +
    ".,;:!?-—'\"()[]{}<>@#$%^&*/\\|+=~`_"
)
VOCAB = {c: i + 1 for i, c in enumerate(CHARS)}  # 0 = PAD
VOCAB_SIZE = len(VOCAB) + 1

MAX_LEN = 1000  # chars
BATCH = 32
EPOCHS = 4
LR = 2e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL2ID = {"Human": 0, "AI": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def pick_dataset() -> pd.DataFrame:
    for p in [DATA_DIR / "train_full.csv", DATA_DIR / "dataset.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        raise FileNotFoundError("No dataset found.")

    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len().between(30, 2000)]
    df = df[df["label"].isin(LABEL2ID.keys())]
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df


def encode_text(s: str, max_len: int = MAX_LEN) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int64)
    s = s[:max_len]
    for i, ch in enumerate(s):
        arr[i] = VOCAB.get(ch, 0)
    return arr


class CharDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.X = np.stack([encode_text(t) for t in texts])
        self.y = np.array(labels, dtype=np.int64)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i])


class CharCNN(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * (MAX_LEN // 4), 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.emb(x).transpose(1, 2)     # (B, E, L)
        x = self.conv(x)                    # (B, 128, L//4)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


def main():
    df = pick_dataset()
    X_tr, X_va, y_tr, y_va = train_test_split(
        df["text"].tolist(), [LABEL2ID[l] for l in df["label"].tolist()],
        test_size=0.12, random_state=SEED, stratify=df["label"].values
    )
    train_ds = CharDataset(X_tr, y_tr)
    val_ds   = CharDataset(X_va, y_va)

    m = CharCNN(VOCAB_SIZE).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    best_val = 0.0
    for epoch in range(1, EPOCHS + 1):
        m.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}")
        for X, y in pbar:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = m(X)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        # eval
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = m(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"[val] acc={acc:.4f}")
        if acc > best_val:
            best_val = acc
            torch.save(m.state_dict(), OUT_DIR / "model.pt")

    # Save config for inference
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(
            {
                "max_len": MAX_LEN,
                "vocab": VOCAB,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
            },
            f, indent=2
        )
    print(f"[ok] Saved best model to {OUT_DIR / 'model.pt'} (best val acc={best_val:.4f})")


if __name__ == "__main__":
    main()
