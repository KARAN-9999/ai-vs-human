# src/train_sbert_v3.py

import numpy as np
import json
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Paths
EMB_DIR = Path("data/embeddings")
TEXT_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load embeddings
X_train = np.load(EMB_DIR / "train_embeddings.npy")
y_train = np.load(EMB_DIR / "train_labels.npy")
X_val = np.load(EMB_DIR / "val_embeddings.npy")
y_val = np.load(EMB_DIR / "val_labels.npy")
X_test = np.load(EMB_DIR / "test_embeddings.npy")
y_test = np.load(EMB_DIR / "test_labels.npy")

# Load original cleaned text
val_texts = pd.read_csv(TEXT_DIR / "val_clean.csv")["text"].tolist()
test_texts = pd.read_csv(TEXT_DIR / "test_clean.csv")["text"].tolist()

# Hyperparameter search
param_grid = {"C": [0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=3, scoring="f1_weighted", n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Evaluation function
def evaluate(X, y, texts, split_name):
    y_pred = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0)
    }
    errors = [
        {"text": text, "true_label": int(true), "predicted_label": int(pred)}
        for text, true, pred in zip(texts, y, y_pred) if true != pred
    ]
    pd.DataFrame(errors).to_csv(REPORTS_DIR / f"sbert_errors_{split_name}.csv", index=False)
    return metrics

# Run evaluations
metrics = {
    "val": evaluate(X_val, y_val, val_texts, "val"),
    "test": evaluate(X_test, y_test, test_texts, "test"),
    "best_params": grid.best_params_
}

# Save model & metrics
joblib.dump(model, MODELS_DIR / "logreg_sbert_tuned.joblib")
with open(REPORTS_DIR / "sbert_results_tuned.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ SBERT hyperparameter search complete.")
print("Best params:", grid.best_params_)
print(json.dumps(metrics, indent=4))
