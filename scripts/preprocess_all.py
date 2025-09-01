"""
preprocess_all.py
~~~~~~~~~~~~~~~~~~

This script extends the existing Kaggle pre‑processing pipeline to allow
combining an additional custom dataset with the original Kaggle CSV. It
cleans, deduplicates and balances the combined data before writing a
balanced sample along with train/validation/test splits. The
original Kaggle preprocessing (`preprocess_kaggle.py`) is left untouched
so that you can always revert to the Kaggle‑only dataset if desired.

If `data/raw/dataset.csv` is not present the script will fall back to
processing only the Kaggle data.

Key differences compared to the Kaggle script:

* Supports reading a second CSV (``data/raw/dataset.csv``). The file
  must include at least a ``text`` column and either a ``label``
  column (with values ``AI``/``Human`` or similar) or a ``generated``
  column (with values 1/0 or '1'/'0').
* Applies the same text cleaning, word length filtering and
  deduplication to both datasets.
* Balances the final combined dataset by sampling up to
  ``DESIRED_PER_CLASS`` examples per class. You can adjust this value
  according to the size of your data.
* Writes the resulting cleaned and balanced dataset to
  ``data/processed/combined_dataset_cleaned.csv`` and the splits to
  ``train_full.csv``, ``val_full.csv`` and ``test_full.csv`` so as not
  to overwrite the Kaggle splits.

Usage::

    python scripts/preprocess_all.py

Ensure you run this script from the project root so that the relative
paths resolve correctly.
"""

import os
import re
import html
import unicodedata
import hashlib
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Path to the Kaggle CSV
KAGGLE_CSV: Path = Path("data/raw/AI_Human.csv")
# Path to the optional custom CSV provided by the user
CUSTOM_CSV: Path = Path("data/raw/dataset.csv")
# Output directory for cleaned and split files
OUT_DIR: Path = Path("data/processed")
# Number of examples per class to keep. Increase if your combined
# dataset contains more samples and you wish to utilise them. If the
# minimum available per class across the combined dataset is less than
# ``DESIRED_PER_CLASS`` the smaller number will be used instead.
DESIRED_PER_CLASS: int = 5000
# Minimum and maximum word length for a sample to be considered
MIN_WORDS: int = 100
MAX_WORDS: int = 400
SEED: int = 42
# When processing the Kaggle CSV we read in chunks to avoid high
# memory usage. Feel free to adjust based on your resources.
CHUNKSIZE: int = 150_000


# -----------------------------------------------------------------------------
# Text cleaning and hashing helpers
# -----------------------------------------------------------------------------
def clean_text(s: str) -> str:
    """Apply a series of cleaning steps to a string.

    - Remove HTML tags
    - Unescape HTML entities
    - Remove URLs
    - Normalise unicode characters
    - Collapse whitespace

    ``str`` inputs are returned in place; non‑string inputs are
    converted to strings before cleaning.
    """
    if not isinstance(s, str):
        s = str(s)
    # strip HTML tags
    s = re.sub(r'<[^>]+>', ' ', s)
    # unescape HTML entities
    s = html.unescape(s)
    # remove URLs
    s = re.sub(r'http[s]?://\S+|www\.\S+', ' ', s)
    # remove common AI model self‑references
    s = re.sub(r'(?i)as an ai (language )?model[,.:;]?\s*', ' ', s)
    s = re.sub(r'(?i)i am an ai\b', ' ', s)
    # normalise unicode
    s = unicodedata.normalize("NFKC", s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def short_hash(s: str) -> str:
    """Compute a short SHA1 hash of the provided string for deduplication."""
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
def load_kaggle_chunks() -> Tuple[List[pd.DataFrame], dict]:
    """Process the Kaggle CSV in chunks and return cleaned samples.

    The function reads the Kaggle CSV in chunks to avoid loading the
    entire file into memory. Each chunk is cleaned, filtered by word
    length and deduplicated using a short hash. Only the ``text`` and
    ``generated`` columns are retained. The function returns a list of
    DataFrames and a dictionary mapping sample hashes to their labels
    (used to prevent duplicates when merging with the custom dataset).
    """
    if not KAGGLE_CSV.exists():
        raise FileNotFoundError(
            f"Kaggle CSV not found: {KAGGLE_CSV}. Ensure the file exists in "
            "data/raw as 'AI_Human.csv'."
        )

    seen_hashes = {}
    collected = []

    # Examine the header to show which columns are available (for user
    # awareness). We don't use this returned DataFrame except for printing.
    df_head = pd.read_csv(KAGGLE_CSV, nrows=5)
    print("[Kaggle] Columns found:", df_head.columns.tolist())
    usecols = ['text', 'generated']

    # Process each chunk
    for chunk in pd.read_csv(KAGGLE_CSV, usecols=usecols, chunksize=CHUNKSIZE):
        chunk = chunk.rename(columns={'text': 'text', 'generated': 'generated'})
        chunk['text'] = chunk['text'].astype(str).apply(clean_text)
        # word length filter
        chunk['word_count'] = chunk['text'].str.split().str.len()
        chunk = chunk[(chunk['word_count'] >= MIN_WORDS) & (chunk['word_count'] <= MAX_WORDS)]
        # hash for deduplication
        chunk['h'] = chunk['text'].apply(short_hash)
        # drop duplicates across all seen hashes
        mask = ~chunk['h'].isin(seen_hashes)
        new_rows = chunk[mask].copy()
        # update seen hashes with label information
        for h, generated in zip(new_rows['h'], new_rows['generated']):
            seen_hashes[h] = generated
        collected.append(new_rows[['text', 'generated', 'h']])
        # stop early if we have enough samples (2 classes)
        total = sum(c.shape[0] for c in collected)
        if total >= DESIRED_PER_CLASS * 2:
            break
    return collected, seen_hashes


def load_custom_dataset(seen_hashes: dict) -> pd.DataFrame:
    """Load and clean the custom CSV if it exists.

    The custom dataset must contain a ``text`` column. It should
    also contain either a ``label`` column with values such as ``AI``/
    ``Human`` or a ``generated`` column with 1/0 values. Any rows with
    missing or unknown labels will be dropped. Duplicate texts (by
    ``short_hash``) that are already present in ``seen_hashes`` (from
    the Kaggle data) will be removed to avoid double counting.

    Args:
        seen_hashes: mapping of previously seen hashes to their label or
            generated value. This is updated when Kaggle data is
            processed.

    Returns:
        A DataFrame with columns ``text``, ``label`` and ``h``.
    """
    if not CUSTOM_CSV.exists():
        print(f"[Custom] CSV not found: {CUSTOM_CSV}. Skipping custom dataset.")
        return pd.DataFrame(columns=['text', 'label', 'h'])

    df = pd.read_csv(CUSTOM_CSV)
    print("[Custom] Columns found:", df.columns.tolist())
    # Ensure the text column exists
    if 'text' not in df.columns:
        raise ValueError(
            f"Custom dataset {CUSTOM_CSV} must contain a 'text' column."
        )
    df['text'] = df['text'].astype(str).apply(clean_text)
    df['word_count'] = df['text'].str.split().str.len()
    df = df[(df['word_count'] >= MIN_WORDS) & (df['word_count'] <= MAX_WORDS)]
    # Determine the label column
    if 'label' in df.columns:
        # Normalize various representations to AI/Human strings
        def normalize_label(x):
            x_str = str(x).strip().lower()
            if x_str in {'ai', '1', 'true', 'yes'}:
                return 'AI'
            if x_str in {'human', '0', 'false', 'no'}:
                return 'Human'
            return None
        df['label'] = df['label'].apply(normalize_label)
    elif 'generated' in df.columns:
        df['label'] = df['generated'].map({1: 'AI', 0: 'Human', '1': 'AI', '0': 'Human'})
    else:
        raise ValueError(
            f"Custom dataset {CUSTOM_CSV} must contain either a 'label' "
            "column or a 'generated' column."
        )
    # Drop rows with unknown labels
    df = df[df['label'].notna()].copy()
    # Create hashes for deduplication
    df['h'] = df['text'].apply(short_hash)
    # Remove duplicates already seen in Kaggle
    df = df[~df['h'].isin(seen_hashes.keys())]
    return df[['text', 'label', 'h']]


# -----------------------------------------------------------------------------
# Main processing function
# -----------------------------------------------------------------------------
def process_and_sample() -> Path:
    """Process Kaggle and custom datasets, deduplicate and balance them.

    Returns the path to the cleaned combined CSV. The file will be saved as
    ``OUT_DIR / combined_dataset_cleaned.csv``.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load Kaggle in chunks
    kaggle_chunks, seen_hashes = load_kaggle_chunks()
    if not kaggle_chunks:
        raise RuntimeError(
            "No rows collected from Kaggle dataset. Check your file path and "
            "column names."
        )
    df_kaggle = pd.concat(kaggle_chunks, ignore_index=True)
    # Map generated column to label names
    df_kaggle['label'] = df_kaggle['generated'].map({1: 'AI', 0: 'Human', '1': 'AI', '0': 'Human'})
    kaggle_counts = df_kaggle['label'].value_counts().to_dict()
    print("[Kaggle] Counts before balancing:", kaggle_counts)

    # Load custom dataset (if exists)
    df_custom = load_custom_dataset(seen_hashes)
    if not df_custom.empty:
        custom_counts = df_custom['label'].value_counts().to_dict()
        print("[Custom] Counts before balancing:", custom_counts)
    else:
        custom_counts = {}

    # Combine
    df_all = pd.concat([
        df_kaggle[['text', 'label', 'h']],
        df_custom[['text', 'label', 'h']]
    ], ignore_index=True)
    # Shuffle to mix Kaggle and custom samples
    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Balance dataset: limit to DESIRED_PER_CLASS from each class if available
    counts = df_all['label'].value_counts().to_dict()
    n = min(DESIRED_PER_CLASS, counts.get('AI', 0), counts.get('Human', 0))
    sampled = pd.concat([
        df_all[df_all['label'] == 'AI'].head(n),
        df_all[df_all['label'] == 'Human'].head(n)
    ], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
    # Save cleaned combined dataset
    clean_path = OUT_DIR / "combined_dataset_cleaned.csv"
    sampled.to_csv(clean_path, index=False)
    print(f"[Combined] Saved cleaned balanced dataset to {clean_path} ({len(sampled)} rows)")
    return clean_path


def make_splits(clean_csv_path: Path) -> Tuple[Path, Path, Path]:
    """Split the cleaned dataset into train/val/test and save the CSVs.

    The splits are saved to ``train_full.csv``, ``val_full.csv`` and
    ``test_full.csv`` in the OUT_DIR. These names are chosen so as not
    to overwrite the original Kaggle splits.

    Returns a tuple of Paths for the train, val and test CSVs.
    """
    df = pd.read_csv(clean_csv_path)
    if 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError("Cleaned CSV must contain 'text' and 'label' columns.")
    train, rest = train_test_split(df, stratify=df['label'], train_size=0.7, random_state=SEED)
    val, test = train_test_split(rest, stratify=rest['label'], test_size=0.5, random_state=SEED)
    train_path = OUT_DIR / "train_full.csv"
    val_path = OUT_DIR / "val_full.csv"
    test_path = OUT_DIR / "test_full.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    print("[Combined] Saved splits:")
    print("  Train:", len(train), "->", train_path)
    print("  Val:", len(val), "->", val_path)
    print("  Test:", len(test), "->", test_path)
    return train_path, val_path, test_path


if __name__ == "__main__":
    cleaned_csv = process_and_sample()
    make_splits(cleaned_csv)