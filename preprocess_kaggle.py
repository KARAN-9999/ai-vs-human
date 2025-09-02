# scripts/preprocess_kaggle.py
import os
import re
import html
import unicodedata
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# --- CONFIG ---
INPUT_CSV = Path("data/raw/dataset_clean.csv")
OUT_DIR = Path("data/processed")
DESIRED_PER_CLASS = 5000       # set None to keep all (balanced by minority)
MIN_WORDS = 100                # lower to 50 if too few rows survive
MAX_WORDS = 400
SEED = 42

# --- Helpers ---
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"<[^>]+>", " ", s)                  # HTML tags
    s = html.unescape(s)
    s = re.sub(r"http[s]?://\S+|www\.\S+", " ", s)  # URLs
    s = re.sub(r"(?i)as an ai (language )?model[,.:;]?\s*", " ", s)
    s = re.sub(r"(?i)\bi am an ai\b", " ", s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

def normalize_label(x) -> Optional[str]:
    """Map anything to exact 'AI' or 'Human'."""
    s = str(x).strip().lower()
    if s in {"ai", "gpt", "machine", "1", "yes", "true"}:
        return "AI"
    if s in {"human", "0", "no", "false"}:
        return "Human"
    # also handle 'Ai', 'aI', 'HUMAN', etc.
    if s == "ai" or s == "aI" or s == "Ai" or s == "AI".lower():
        return "AI"
    if s == "human".lower() or s == "Human".lower():
        return "Human"
    return None

def process_and_sample():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV, usecols=["text", "label"])
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["word_count"] = df["text"].str.split().str.len()
    df = df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)]

    # dedup
    df["h"] = df["text"].apply(short_hash)
    df = df.drop_duplicates("h")

    # normalize label values strictly to 'AI' / 'Human'
    df["label"] = df["label"].apply(normalize_label)
    df = df[df["label"].isin(["AI", "Human"])]

    counts = df["label"].value_counts().to_dict()
    print("Counts before balancing:", counts)

    if not counts.get("AI") or not counts.get("Human"):
        raise RuntimeError(
            "One class is missing after cleaning. "
            "Tip: print a few unique raw labels, check MIN/MAX_WORDS, or inspect the CSV."
        )

    # decide per-class sample size
    if DESIRED_PER_CLASS is None:
        n = min(counts["AI"], counts["Human"])
    else:
        n = min(DESIRED_PER_CLASS, counts["AI"], counts["Human"])

    if n < 1:
        raise RuntimeError(
            f"Per-class sample size computed as {n}. "
            "Set DESIRED_PER_CLASS=None or lower MIN_WORDS to keep more rows."
        )

    df_bal = pd.concat(
        [
            df[df["label"] == "AI"].sample(n, random_state=SEED),
            df[df["label"] == "Human"].sample(n, random_state=SEED),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=SEED).reset_index(drop=True)

    clean_path = OUT_DIR / "dataset_cleaned_sample.csv"
    df_bal.to_csv(clean_path, index=False)
    print(f"Saved balanced dataset: {clean_path} ({len(df_bal)} rows)")
    return clean_path

def make_splits(clean_csv_path: Path):
    df = pd.read_csv(clean_csv_path)
    train, rest = train_test_split(df, stratify=df["label"], train_size=0.7, random_state=SEED)
    val, test = train_test_split(rest, stratify=rest["label"], test_size=0.5, random_state=SEED)

    # Write files properly (avoid write_text on CSV strings)
    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)

    print("Saved splits:")
    print("  Train:", len(train))
    print("  Val:  ", len(val))
    print("  Test: ", len(test))

if __name__ == "__main__":
    clean_csv = process_and_sample()
    make_splits(clean_csv)
