# preprocess_kaggle.py
import os
import re
import html
import unicodedata
import hashlib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# CONFIG
INPUT_CSV = Path("data/raw/AI_Human.csv")
OUT_DIR = Path("data/processed")
DESIRED_PER_CLASS = 5000   # how many AI and Human samples to keep
MIN_WORDS = 100
MAX_WORDS = 400
SEED = 42
CHUNKSIZE = 150_000

# Clean text
def clean_text(s):
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r'<[^>]+>', ' ', s)  # HTML tags
    s = html.unescape(s)
    s = re.sub(r'http[s]?://\S+|www\.\S+', ' ', s)  # URLs
    s = re.sub(r'(?i)as an ai (language )?model[,.:;]?\s*', ' ', s)
    s = re.sub(r'(?i)i am an ai\b', ' ', s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Hash for deduplication
def short_hash(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]

def process_and_sample():
    os.makedirs(OUT_DIR, exist_ok=True)
    seen_hashes = set()
    collected = []

    print("Reading and cleaning CSV in chunks...")
    df_head = pd.read_csv(INPUT_CSV, nrows=5)
    print("Columns found:", df_head.columns.tolist())

    usecols = ['text', 'generated']

    for chunk in pd.read_csv(INPUT_CSV, usecols=usecols, chunksize=CHUNKSIZE):
        chunk = chunk.rename(columns={'text': 'text', 'generated': 'generated'})
        chunk['text'] = chunk['text'].astype(str).apply(clean_text)
        chunk['word_count'] = chunk['text'].str.split().str.len()
        chunk = chunk[(chunk['word_count'] >= MIN_WORDS) & (chunk['word_count'] <= MAX_WORDS)]
        chunk['h'] = chunk['text'].apply(short_hash)
        chunk = chunk[~chunk['h'].isin(seen_hashes)]
        seen_hashes.update(chunk['h'].tolist())
        collected.append(chunk[['text', 'generated']])

        total_collected = sum(c.shape[0] for c in collected)
        if total_collected >= DESIRED_PER_CLASS * 2:
            print("Enough examples collected. Stopping.")
            break

    if not collected:
        raise RuntimeError("No rows collected - check CSV path and column names.")

    df_all = pd.concat(collected, ignore_index=True)
    df_all['label'] = df_all['generated'].map({1: 'AI', 0: 'Human', '1': 'AI', '0': 'Human'})

    counts = df_all['label'].value_counts().to_dict()
    print("Counts before balancing:", counts)
    n = min(DESIRED_PER_CLASS, counts.get('AI',0), counts.get('Human',0))
    sampled = pd.concat([
        df_all[df_all['label']=='AI'].sample(n, random_state=SEED),
        df_all[df_all['label']=='Human'].sample(n, random_state=SEED)
    ], ignore_index=True)

    sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)
    clean_path = OUT_DIR / "dataset_cleaned_sample.csv"
    sampled.to_csv(clean_path, index=False)
    print(f"Saved cleaned balanced dataset to {clean_path} ({len(sampled)} rows)")

    return clean_path

def make_splits(clean_csv_path):
    df = pd.read_csv(clean_csv_path)
    train, rest = train_test_split(df, stratify=df['label'], train_size=0.7, random_state=SEED)
    val, test = train_test_split(rest, stratify=rest['label'], test_size=0.5, random_state=SEED)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    val.to_csv(OUT_DIR / "val.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print("Saved splits:")
    print(" Train:", len(train))
    print(" Val:", len(val))
    print(" Test:", len(test))

if __name__ == "__main__":
    clean_csv = process_and_sample()
    make_splits(clean_csv)
