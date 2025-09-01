"""
Generic text pre‑processing for AI vs Human datasets.

This module cleans raw text by removing HTML tags, URLs and non‑basic
characters, lowercases the text and discards very short samples. It
contains a convenience ``preprocess_file`` function for cleaning a CSV
containing at least a ``text`` column and writing the cleaned version to
disk. When run as a script it will clean both the Kaggle splits
(``train.csv``, ``val.csv``, ``test.csv``) and any combined dataset splits
(``train_full.csv``, ``val_full.csv``, ``test_full.csv``) if they are
present. Cleaned files are named with a ``_clean.csv`` suffix and
written to the same directory.

This file is a local copy of the pre‑processing script from the
upstream repository with additional logic to handle combined datasets.
"""

import re
import html
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Paths
ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"


def clean_text(s: str) -> str:
    """Clean an input string.

    The cleaning pipeline removes HTML tags, unescapes HTML entities,
    replaces newlines with spaces, normalises unicode characters,
    converts to lowercase, retains basic punctuation and collapses
    multiple spaces.

    Args:
        s: The input value. If not a string it will be cast to one.

    Returns:
        A cleaned lowercased string.
    """
    if not isinstance(s, str):
        s = str(s)
    s = html.unescape(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"<[^>]+>", " ", s)  # remove HTML tags
    s = re.sub(r"http\S+", " ", s)  # remove URLs
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\.\,\?\!']", " ", s)  # keep basic punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def preprocess_file(in_path: Path, out_path: Path) -> pd.DataFrame:
    """Load a CSV, clean its text and write it back out.

    The CSV must contain a ``text`` column. Rows with fewer than 50 words
    (after cleaning) are dropped. Duplicate ``text`` rows are removed.

    Args:
        in_path: Path to the input CSV.
        out_path: Path where the cleaned CSV will be saved.

    Returns:
        The cleaned DataFrame.
    """
    if not in_path.exists():
        print(f"[skip] {in_path} does not exist; skipping")
        return pd.DataFrame()
    df = pd.read_csv(in_path)
    if 'text' not in df.columns:
        raise ValueError(f"Input file {in_path} must contain a 'text' column.")
    tqdm.pandas(desc=f"Cleaning {in_path.name}")
    df['text'] = df['text'].progress_apply(clean_text)
    df['length'] = df['text'].str.split().str.len()
    df = df[df['length'] >= 50]  # remove very short texts
    df = df.dropna(subset=['text']).drop_duplicates(subset=['text']).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[clean] Saved cleaned file: {out_path} (rows={len(df)})")
    return df


if __name__ == "__main__":
    """Entry point for CLI usage.

    When invoked directly this script will attempt to clean the standard
    Kaggle dataset splits as well as the combined splits (if present)
    produced by ``scripts/preprocess_all.py``. Each existing CSV will be
    cleaned to produce a corresponding ``*_clean.csv``. Missing files
    will be skipped and reported.
    """
    # Clean Kaggle splits
    for name in ["train", "val", "test"]:
        in_path = PROCESSED / f"{name}.csv"
        out_path = PROCESSED / f"{name}_clean.csv"
        preprocess_file(in_path, out_path)
    # Clean combined splits if present
    for name in ["train_full", "val_full", "test_full"]:
        in_path = PROCESSED / f"{name}.csv"
        out_path = PROCESSED / f"{name}_clean.csv"
        preprocess_file(in_path, out_path)