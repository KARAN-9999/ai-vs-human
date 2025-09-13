import pandas as pd
from pathlib import Path

DATA = Path("data/raw")
DATA.mkdir(exist_ok=True)

# Edit these paths to whatever you actually have:
sources = [
    DATA/"AI_Human.csv",         # Kaggle (existing)
    DATA/"dataset.csv",   
    DATA /"augmented_human.csv",
    DATA/"augmented_ai.csv"
              # your own collected data
    # add more if you have them
]

dfs = []
for p in sources:
    if p.exists():
        df = pd.read_csv(p)
        # normalize columns
        cols = {c.lower().strip(): c for c in df.columns}
        text_col = [c for c in df.columns if c.lower() in ("text","content","body")]
        label_col = [c for c in df.columns if c.lower() in ("label","target","class")]
        if not text_col or not label_col:
            continue
        df = df[[text_col[0], label_col[0]]].rename(columns={text_col[0]:"text", label_col[0]:"label"})
        # normalize labels to AI/Human
        df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ai":"AI","human":"Human"})
        df = df.dropna(subset=["text","label"])
        dfs.append(df)

assert dfs, "No input datasets found. Put csvs in data/ first."
full = pd.concat(dfs, ignore_index=True)

# dedupe & shuffle
full["text"] = full["text"].astype(str).str.strip()
full = full.drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=42)

# tiny cleanups: remove ultra-short texts
full = full[full["text"].str.len() >= 20]

out = DATA/"dataset_clean.csv"
full.to_csv(out, index=False)
print("Wrote:", out, "rows:", len(full))
