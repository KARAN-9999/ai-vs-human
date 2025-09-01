"""
Transformer embedding extraction and classifier training (v2).

This script is adapted from the original repository to automatically
detect whether the combined dataset produced by
``scripts/preprocess_all.py`` is available. It reads the cleaned train,
validation and test splits from ``data/processed`` and extracts
transformer embeddings using a HuggingFace model (default:
``distilroberta-base``). A Logistic Regression classifier is trained
with a small grid search over the regularisation parameter C. The best
model and embeddings are saved alongside metrics, error samples and
plots. If combined splits (``*_full_clean.csv``) exist, they are used;
otherwise the Kaggle splits (``*_clean.csv``) are loaded.

Note: This script requires the ``transformers`` and ``torch`` packages.
Run ``pip install transformers torch`` if they are not installed.
"""

import os
import json
from pathlib import Path
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
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)


# ------------------------
# Config / paths
# ------------------------
MODEL_NAME = "distilroberta-base"
MAX_LEN = 256
BATCH_SIZE = 32

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "processed"
EMB_DIR = BASE_DIR / "data" / "embeddings"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

EMB_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter grid for LogisticRegression
C_GRID = [0.01, 0.1, 1, 10]


# ------------------------
# Helpers
# ------------------------
def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """Apply mean pooling on the last hidden states using the attention mask."""
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()


def get_embeddings(
    texts: list[str], tokenizer: AutoTokenizer, model: AutoModel, device: torch.device, batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """Extract embeddings for a list of texts using batching."""
    model.eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
            batch_texts = texts[i : i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden = out.last_hidden_state
            batch_emb = mean_pooling(last_hidden, attention_mask)
            embeddings.append(batch_emb)
    return np.vstack(embeddings)


def ensure_embeddings_exist(
    split: str, tokenizer: AutoTokenizer, model: AutoModel,
    device: torch.device, texts: list[str]
) -> np.ndarray:
    """Load embeddings from disk if they exist and match dataset size,
    otherwise extract and save them."""
    emb_path = EMB_DIR / f"transformer_{split}_embeddings.npy"
    # If file exists, try to load and verify shape
    if emb_path.exists():
        arr = np.load(emb_path)
        if arr.shape[0] == len(texts):
            print(f"[skip] embeddings exist: {emb_path}")
            return arr
        else:
            print(f"[recompute] {emb_path} shape {arr.shape} doesn’t match "
                  f"{len(texts)} samples; re-extracting...")
    # Compute and save new embeddings
    print(f"[create] extracting embeddings for {split} (this may take a while)...")
    arr = get_embeddings(texts, tokenizer, model, device)
    np.save(emb_path, arr)
    print(f"Saved embeddings: {emb_path} (shape={arr.shape})")
    return arr



def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test DataFrames, preferring combined splits if present."""
    train_full = DATA_DIR / "train_full_clean.csv"
    val_full = DATA_DIR / "val_full_clean.csv"
    test_full = DATA_DIR / "test_full_clean.csv"
    if train_full.exists() and val_full.exists() and test_full.exists():
        print("[data] Using combined dataset for transformer training.")
        train_df = pd.read_csv(train_full)
        val_df = pd.read_csv(val_full)
        test_df = pd.read_csv(test_full)
        return train_df, val_df, test_df
    # Fallback to Kaggle splits
    train_clean = DATA_DIR / "train_clean.csv"
    val_clean = DATA_DIR / "val_clean.csv"
    test_clean = DATA_DIR / "test_clean.csv"
    if not (train_clean.exists() and val_clean.exists() and test_clean.exists()):
        raise FileNotFoundError(
            "Could not find any cleaned dataset splits. Run src/preprocess.py first."
        )
    print("[data] Using Kaggle dataset for transformer training.")
    return (
        pd.read_csv(train_clean),
        pd.read_csv(val_clean),
        pd.read_csv(test_clean),
    )


def main() -> None:
    # Load the splits
    train_df, val_df, test_df = load_splits()
    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    # Device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # Extract or load embeddings
    X_train = ensure_embeddings_exist("train", tokenizer, model, device, train_texts)
    X_val = ensure_embeddings_exist("val", tokenizer, model, device, val_texts)
    X_test = ensure_embeddings_exist("test", tokenizer, model, device, test_texts)

    # Label encoding
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    y_val = le.transform(val_df["label"].astype(str))
    y_test = le.transform(test_df["label"].astype(str))
    joblib.dump(le, MODELS_DIR / "label_encoder_transformer.joblib")
    print("Label classes:", list(le.classes_))
    if len(le.classes_) > 1:
        print(f"Positive class label=1 -> {le.inverse_transform([1])[0]}")

    # Grid search over C to pick best model on validation F1
    best_C = None
    best_f1 = -1.0
    best_model: LogisticRegression | None = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2_000)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
        print(f"C={C} -> val F1={f1:.6f}")
        if f1 > best_f1:
            best_f1 = f1
            best_C = C
            best_model = clf
    print(f"Selected best_C={best_C} with val F1={best_f1:.6f}")

    # Evaluate best model
    def eval_split(clf: LogisticRegression, X: np.ndarray, y: np.ndarray, texts: list[str], split_name: str):
        y_pred = clf.predict(X)
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        }
        # Build error DataFrame with prediction probabilities for ROC curves
        probs = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None
        errors = []
        for i, (txt, true_lbl, pred_lbl) in enumerate(zip(texts, y, y_pred)):
            if true_lbl != pred_lbl:
                rec = {
                    "index": int(i),
                    "text": txt,
                    "true_label": int(true_lbl),
                    "predicted_label": int(pred_lbl),
                }
                if probs is not None:
                    rec["pred_proba"] = float(probs[i].max())
                errors.append(rec)
        errors_df = pd.DataFrame(errors)
        return metrics, errors_df, probs, y_pred

    val_metrics, val_errors_df, val_probs, val_preds = eval_split(best_model, X_val, y_val, val_texts, "val")
    test_metrics, test_errors_df, test_probs, test_preds = eval_split(best_model, X_test, y_test, test_texts, "test")

    # Save full predictions (useful for ROC curves)
    if val_probs is not None:
        val_full_df = pd.DataFrame({
            "text": val_texts,
            "true_label": y_val,
            "predicted_label": val_preds,
            "pred_proba": val_probs[:, 1] if val_probs.shape[1] > 1 else val_probs[:, 0],
        })
        val_full_df.to_csv(REPORTS_DIR / "transformer_errors_val.csv", index=False)
    if test_probs is not None:
        test_full_df = pd.DataFrame({
            "text": test_texts,
            "true_label": y_test,
            "predicted_label": test_preds,
            "pred_proba": test_probs[:, 1] if test_probs.shape[1] > 1 else test_probs[:, 0],
        })
        test_full_df.to_csv(REPORTS_DIR / "transformer_errors_test.csv", index=False)

    # Save model and embeddings
    joblib.dump(best_model, MODELS_DIR / "logreg_transformer_emb_best.joblib")
    np.save(EMB_DIR / "transformer_train_embeddings.npy", X_train)
    np.save(EMB_DIR / "transformer_val_embeddings.npy", X_val)
    np.save(EMB_DIR / "transformer_test_embeddings.npy", X_test)

    # Build results dictionary
    results = {
        "best_C": best_C,
        "label_classes": list(le.classes_),
        "val": val_metrics,
        "test": test_metrics,
    }

    # Save metrics JSON (v2)
    results_path = REPORTS_DIR / "transformer_emb_results_v2.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save error samples (top 50 based on probability)
    if not val_errors_df.empty:
        val_errors_df.sort_values(by="pred_proba", ascending=False, inplace=True, ignore_index=True)
        val_errors_df.head(50).to_csv(REPORTS_DIR / "transformer_errors_val_top50.csv", index=False)
    if not test_errors_df.empty:
        test_errors_df.sort_values(by="pred_proba", ascending=False, inplace=True, ignore_index=True)
        test_errors_df.head(50).to_csv(REPORTS_DIR / "transformer_errors_test_top50.csv", index=False)

    # Confusion matrices
    def plot_cm(y_true: np.ndarray, y_pred: np.ndarray, split_name: str) -> None:
        labels = list(le.classes_)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - Transformer Emb - {split_name}")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / f"cm_transformer_{split_name}.png")
        plt.close()

    plot_cm(y_val, val_preds, "val")
    plot_cm(y_test, test_preds, "test")

    # ROC curves (binary only)
    if len(le.classes_) == 2 and val_probs is not None and test_probs is not None:
        pos = 1  # use label index 1 as the positive class
        def plot_roc(y_true: np.ndarray, probs: np.ndarray, split_name: str) -> float:
            fpr, tpr, _ = roc_curve(y_true, probs[:, pos], pos_label=pos)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - Transformer Emb - {split_name}")
            plt.legend(loc="lower right")
            plt.savefig(REPORTS_DIR / f"roc_transformer_{split_name}.png")
            plt.close()
            return roc_auc

        roc_val = plot_roc(y_val, val_probs, "val")
        roc_test = plot_roc(y_test, test_probs, "test")
        results["val"]["roc_auc"] = float(roc_val)
        results["test"]["roc_auc"] = float(roc_test)
        # Save updated results with ROC values
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)

    # Summary markdown
    md_lines: list[str] = []
    md_lines.append("# Transformer Embeddings - Summary (v2)\n")
    md_lines.append("## Model & Setup\n")
    md_lines.append(f"- Encoder: {MODEL_NAME}\n")
    md_lines.append(f"- LogisticRegression grid C: {C_GRID}\n")
    md_lines.append(f"- Selected best_C: {best_C}\n")
    md_lines.append(f"- Device used: {device}\n")
    md_lines.append("\n## Label mapping\n")
    for idx, name in enumerate(le.classes_):
        md_lines.append(f"- {idx} => {name}\n")
    md_lines.append("\n## Validation metrics\n")
    for k, v in val_metrics.items():
        md_lines.append(f"- {k}: {v:.4f}\n")
    md_lines.append("\n## Test metrics\n")
    for k, v in test_metrics.items():
        md_lines.append(f"- {k}: {v:.4f}\n")
    if "roc_auc" in results["val"]:
        md_lines.append(f"\n- val ROC AUC: {results['val']['roc_auc']:.4f}\n")
        md_lines.append(f"- test ROC AUC: {results['test']['roc_auc']:.4f}\n")
    md_path = REPORTS_DIR / "transformer_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    print("✅ Transformer embedding training (v2) complete.")
    print("Metrics saved to:", results_path)
    print("Summary markdown:", md_path)


if __name__ == "__main__":
    main()