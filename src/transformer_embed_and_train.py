# src/transformer_embed_and_train.py
"""
Extract transformer embeddings (mean pooling) and train a LogisticRegression classifier.
Saves model, metrics, error samples, and basic plots to reports/.
"""

import os
from pathlib import Path
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# ------------------------
# Config / paths
# ------------------------
MODEL_NAME = "distilroberta-base"   # change if you want another encoder
MAX_LEN = 256
BATCH_SIZE = 32                     # reduce if memory issues
EMB_DIR = Path("data/embeddings")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")

EMB_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Utility: embedding extractor
# ------------------------
def mean_pooling(hidden_states, attention_mask):
    """
    hidden_states: torch.Tensor (batch, seq_len, hidden)
    attention_mask: torch.Tensor (batch, seq_len)
    returns: numpy array (batch, hidden)
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (batch, seq_len, 1)
    masked = hidden_states * mask
    summed = masked.sum(dim=1)           # (batch, hidden)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # avoid div by zero
    return (summed / counts).cpu().numpy()

def get_embeddings(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i: i + batch_size]
            enc = tokenizer(batch_texts,
                            padding="longest",
                            truncation=True,
                            max_length=MAX_LEN,
                            return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = out.last_hidden_state  # (batch, seq_len, hidden)
            batch_emb = mean_pooling(last_hidden, attention_mask)  # (batch, hidden)
            embeddings.append(batch_emb)
    return np.vstack(embeddings)

# ------------------------
# Main
# ------------------------
def main():
    # load processed CSVs
    train_df = pd.read_csv("data/processed/train_clean.csv")
    val_df = pd.read_csv("data/processed/val_clean.csv")
    test_df = pd.read_csv("data/processed/test_clean.csv")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # get texts
    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    # Extract embeddings (will reuse / save)
    print("Extracting train embeddings...")
    X_train = get_embeddings(train_texts, tokenizer, model, device)
    print("Extracting val embeddings...")
    X_val = get_embeddings(val_texts, tokenizer, model, device)
    print("Extracting test embeddings...")
    X_test = get_embeddings(test_texts, tokenizer, model, device)

    # Save embeddings for reuse
    np.save(EMB_DIR / "transformer_train_embeddings.npy", X_train)
    np.save(EMB_DIR / "transformer_val_embeddings.npy", X_val)
    np.save(EMB_DIR / "transformer_test_embeddings.npy", X_test)

    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    y_val = le.transform(val_df["label"].astype(str))
    y_test = le.transform(test_df["label"].astype(str))
    joblib.dump(le, MODELS_DIR / "label_encoder_transformer.joblib")

    # Train a simple classifier
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    # Predict & evaluate
    def evaluate_and_save(X, y, texts, split_name):
        y_pred = clf.predict(X)
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0))
        }
        # error samples
        errors = []
        for t, yt, yp in zip(texts, y, y_pred):
            if yt != yp:
                errors.append({"text": t, "true_label": int(yt), "predicted_label": int(yp)})
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(REPORTS_DIR / f"transformer_errors_{split_name}.csv", index=False)
        return metrics, errors_df, y_pred

    val_metrics, val_errors, y_val_pred = evaluate_and_save(X_val, y_val, val_texts, "val")
    test_metrics, test_errors, y_test_pred = evaluate_and_save(X_test, y_test, test_texts, "test")

    # Save classifier + metrics
    joblib.dump(clf, MODELS_DIR / "logreg_transformer_emb.joblib")
    with open(REPORTS_DIR / "transformer_emb_results.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=4)

    print("Metrics (val):", val_metrics)
    print("Metrics (test):", test_metrics)

    # Confusion matrix plots
    def plot_cm(y_true, y_pred, split_name):
        labels = le.classes_
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - Transformer Embeddings - {split_name}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"cm_transformer_{split_name}.png")
        plt.close()

    plot_cm(y_val, y_val_pred, "val")
    plot_cm(y_test, y_test_pred, "test")

    # ROC curves (binary only)
    if len(le.classes_) == 2:
        pos_label = 1  # by label encoder ordering; if you want 'AI' positive, decode and choose accordingly
        def plot_roc(y_true, y_score, split_name):
            fpr, tpr, _ = roc_curve(y_true, y_score[:, pos_label], pos_label=pos_label)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
            plt.plot([0,1],[0,1],"k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - Transformer Embeddings - {split_name}")
            plt.legend()
            plt.savefig(REPORTS_DIR / f"roc_transformer_{split_name}.png")
            plt.close()

        # need probability scores
        if hasattr(clf, "predict_proba"):
            plot_roc(y_val, clf.predict_proba(X_val), "val")
            plot_roc(y_test, clf.predict_proba(X_test), "test")

    print("✅ Done. Embeddings and results saved to:")
    print(" -", EMB_DIR)
    print(" -", MODELS_DIR)
    print(" -", REPORTS_DIR)

if __name__ == "__main__":
    main()
