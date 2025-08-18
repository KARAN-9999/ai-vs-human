import re
import html
import unicodedata
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Paths
ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = html.unescape(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r'<[^>]+>', ' ', s)                 # remove HTML tags
    s = re.sub(r'http\S+', ' ', s)                 # remove URLs
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\.\,\?\!\']", " ", s)   # keep basic punctuation
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def preprocess_file(in_path: Path, out_path: Path):
    df = pd.read_csv(in_path)
    tqdm.pandas(desc=f"Cleaning {in_path.name}")
    df['text'] = df['text'].progress_apply(clean_text)
    df['length'] = df['text'].str.split().str.len()
    df = df[df['length'] >= 50]    # remove very short texts
    df = df.dropna(subset=['text']).drop_duplicates(subset=['text'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved cleaned file: {out_path} (rows={len(df)})")
    return df

if __name__ == "__main__":
    train = preprocess_file(PROCESSED / "train.csv", PROCESSED / "train_clean.csv")
    val   = preprocess_file(PROCESSED / "val.csv",   PROCESSED / "val_clean.csv")
    test  = preprocess_file(PROCESSED / "test.csv",  PROCESSED / "test_clean.csv")
