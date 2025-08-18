# src/train_sbert_v1.py

import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Paths
EMB_DIR = Path("data/embeddings")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load embeddings (allow_pickle=True for string/object labels)
X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy", allow_pickle=True)
X_val = np.load(EMB_DIR / "val_embeddings.npy")
y_val = np.load(EMB_DIR / "val_labels.npy", allow_pickle=True)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)

# Metrics
metrics = {
    "accuracy": accuracy_score(y_val, y_pred),
    "precision": precision_score(y_val, y_pred, average="weighted", zero_division=0),
    "recall": recall_score(y_val, y_pred, average="weighted", zero_division=0),
    "f1": f1_score(y_val, y_pred, average="weighted", zero_division=0)
}

# Save model & metrics
joblib.dump(model, MODELS_DIR / "logreg_sbert.joblib")
with open(REPORTS_DIR / "sbert_results.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ SBERT baseline complete. Metrics saved to reports/sbert_results.json")
