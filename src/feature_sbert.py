# src/feature_sbert.py

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# Paths
DATA_DIR = Path("data/processed")
EMB_DIR = Path("data/embeddings")
MODELS_DIR = Path("models")
EMB_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

# Encode labels into integers
label_encoder = LabelEncoder()
label_encoder.fit(train_df["label"])
joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

def encode_and_save(df, split):
    embeddings = model.encode(
        df["text"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )
    labels = label_encoder.transform(df["label"])
    np.save(EMB_DIR / f"{split}_embeddings.npy", embeddings)
    np.save(EMB_DIR / f"{split}_labels.npy", labels)
    print(f"✅ Saved {split} embeddings and labels")

# Encode all splits
encode_and_save(train_df, "train")
encode_and_save(val_df, "val")
encode_and_save(test_df, "test")

print("✅ All embeddings & label encoder saved.")
