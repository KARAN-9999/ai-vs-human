# src/train_transformer_emb_v2.py
"""
train_transformer_emb_v2.py

- Extracts transformer embeddings (mean pooling) for train/val/test (if not present).
- Trains LogisticRegression with a small grid search on C values using validation F1 to pick best.
- Auto-detects label encoding; prints which class is label=1 (positive).
- Saves: embeddings (.npy), best model (.joblib), label encoder, metrics JSON, top-10 misclassified CSV,
  confusion matrices and ROC plots, and a transformer_summary.md in reports/.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

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

# Grid for LR
C_GRID = [0.01, 0.1, 1, 10]

# ------------------------
# Helpers
# ------------------------
def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
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
            last_hidden = out.last_hidden_state
            batch_emb = mean_pooling(last_hidden, attention_mask)
            embeddings.append(batch_emb)
    return np.vstack(embeddings)

def save_metrics_json(path: Path, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def ensure_embeddings_exist(split: str, tokenizer, model, device, df_texts):
    emb_path = EMB_DIR / f"transformer_{split}_embeddings.npy"
    if emb_path.exists():
        print(f"[skip] embeddings exist: {emb_path}")
        return np.load(emb_path)
    print(f"[create] extracting embeddings for {split} (this may take a while)...")
    arr = get_embeddings(df_texts, tokenizer, model, device)
    np.save(emb_path, arr)
    print(f"Saved embeddings: {emb_path} (shape={arr.shape})")
    return arr

# ------------------------
# Main pipeline
# ------------------------
def main():
    # Load CSVs
    train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
    val_df = pd.read_csv(DATA_DIR / "val_clean.csv")
    test_df = pd.read_csv(DATA_DIR / "test_clean.csv")

    train_texts = train_df["text"].astype(str).tolist()
    val_texts = val_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    # device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # Ensure embeddings exist (or create them)
    X_train = ensure_embeddings_exist("train", tokenizer, model, device, train_texts)
    X_val = ensure_embeddings_exist("val", tokenizer, model, device, val_texts)
    X_test = ensure_embeddings_exist("test", tokenizer, model, device, test_texts)

    # Label encode
    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str))
    y_val = le.transform(val_df["label"].astype(str))
    y_test = le.transform(test_df["label"].astype(str))

    joblib.dump(le, MODELS_DIR / "label_encoder_transformer.joblib")
    label_mapping = {int(i): int(i) for i in range(len(le.classes_))}
    # also save mapping name->int
    mapping_name_to_int = {name: int(idx) for idx, name in enumerate(le.classes_)}
    print("Label classes:", list(le.classes_))
    # which class is label==1?
    if 1 < len(le.classes_):
        print(f"Positive class label=1 -> {le.inverse_transform([1])[0]}")
    else:
        print("Warning: only one class found in label encoder!")

    # Grid search on C (train -> val)
    best_C = None
    best_f1 = -1.0
    best_model = None

    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        f1 = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)
        print(f"C={C} -> val F1={f1:.6f}")
        if f1 > best_f1:
            best_f1 = f1
            best_C = C
            best_model = clf

    print(f"Selected best_C={best_C} with val F1={best_f1:.6f}")

    # Evaluate best model on val & test
    def eval_split(clf, X, y, texts, split_name):
        y_pred = clf.predict(X)
        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0))
        }
        # top misclassified
        errors = []
        probs = None
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)
        for i, (txt, true, pred) in enumerate(zip(texts, y, y_pred)):
            if true != pred:
                rec = {
                    "index": int(i),
                    "text": txt,
                    "true_label": int(true),
                    "predicted_label": int(pred)
                }
                if probs is not None:
                    rec["pred_proba"] = float(probs[i].max())
                errors.append(rec)
        # return metrics and errors df
        errors_df = pd.DataFrame(errors)
        return metrics, errors_df, probs, y_pred

    val_metrics, val_errors_df, val_probs, val_preds = eval_split(best_model, X_val, y_val, val_texts, "val")
    test_metrics, test_errors_df, test_probs, test_preds = eval_split(best_model, X_test, y_test, test_texts, "test")
    # Save full predictions (with probabilities for ROC)
    if val_probs is not None:
        val_full_df = pd.DataFrame({
        "text": val_texts,
        "true_label": y_val,
        "predicted_label": val_preds,
        "pred_proba": val_probs[:, 1]  # prob of positive class
    })
    val_full_df.to_csv(REPORTS_DIR / "transformer_errors_val.csv", index=False)

    if test_probs is not None:
        test_full_df = pd.DataFrame({
        "text": test_texts,
        "true_label": y_test,
        "predicted_label": test_preds,
        "pred_proba": test_probs[:, 1]  # prob of positive class
    })
    test_full_df.to_csv(REPORTS_DIR / "transformer_errors_test.csv", index=False)


    # Save best model and embeddings
    joblib.dump(best_model, MODELS_DIR / "logreg_transformer_emb_best.joblib")
    np.save(EMB_DIR / "transformer_train_embeddings.npy", X_train)
    np.save(EMB_DIR / "transformer_val_embeddings.npy", X_val)
    np.save(EMB_DIR / "transformer_test_embeddings.npy", X_test)

    # Save metrics + meta
    results = {
        "best_C": best_C,
        "label_classes": list(le.classes_),
        "val": val_metrics,
        "test": test_metrics
    }
    save_path = REPORTS_DIR / "transformer_emb_results_v2.json"
    save_metrics_json(save_path, results)

    # Save error CSVs (top 10 misclassified)
    val_errors_df.sort_values(by="pred_proba", ascending=False, inplace=True, ignore_index=True)
    test_errors_df.sort_values(by="pred_proba", ascending=False, inplace=True, ignore_index=True)
    val_errors_df.head(50).to_csv(REPORTS_DIR / "transformer_errors_val_top50.csv", index=False)
    test_errors_df.head(50).to_csv(REPORTS_DIR / "transformer_errors_test_top50.csv", index=False)

    # Confusion matrices
    def plot_cm(y_true, y_pred, split_name):
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
        pos_label_name = le.inverse_transform([1])[0] if len(le.classes_) > 1 else None
        # get numeric arrays for ROC (use label index 1 as positive)
        def plot_roc(y_true, probs, split_name):
            # probs shape (n, n_classes)
            # compute roc for pos class index 1
            pos = 1
            fpr, tpr, _ = roc_curve(y_true, probs[:, pos], pos_label=pos)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - Transformer Emb - {split_name} (pos={pos_label_name})")
            plt.legend(loc="lower right")
            plt.savefig(REPORTS_DIR / f"roc_transformer_{split_name}.png")
            plt.close()
            return roc_auc

        roc_val = plot_roc(y_val, val_probs, "val")
        roc_test = plot_roc(y_test, test_probs, "test")
        results["val"]["roc_auc"] = float(roc_val)
        results["test"]["roc_auc"] = float(roc_test)

    # Save final results with ROC if present
    save_metrics_json(save_path, results)

    # Generate summary markdown
    md = []
    md.append("# Transformer Embeddings - Summary (v2)\n")
    md.append("## Model & Setup\n")
    md.append(f"- Encoder: {MODEL_NAME}\n")
    md.append(f"- LogisticRegression grid C: {C_GRID}\n")
    md.append(f"- Selected best_C: {best_C}\n")
    md.append(f"- Device used: {device}\n")
    md.append("\n## Label mapping\n")
    for idx, name in enumerate(le.classes_):
        md.append(f"- {idx} => {name}\n")
    md.append("\n## Validation metrics\n")
    for k, v in val_metrics.items():
        md.append(f"- {k}: {v:.4f}\n")
    md.append("\n## Test metrics\n")
    for k, v in test_metrics.items():
        md.append(f"- {k}: {v:.4f}\n")
    if "roc_auc" in results["val"]:
        md.append(f"\n- val ROC AUC: {results['val']['roc_auc']:.4f}\n")
        md.append(f"\n- test ROC AUC: {results['test']['roc_auc']:.4f}\n")

    # include top-10 misclassifications
    md.append("\n## Top misclassified examples (val)\n")
    for i, row in val_errors_df.head(10).iterrows():
        true_label = le.inverse_transform([int(row["true_label"])])[0]
        pred_label = le.inverse_transform([int(row["predicted_label"])])[0]
        md.append(f"\n**Example {i+1}** (true: {true_label}, pred: {pred_label})\n\n")
        text_snip = row["text"]
        if len(text_snip) > 500:
            text_snip = text_snip[:500] + "..."
        md.append(f"{text_snip}\n")

    summary_path = REPORTS_DIR / "transformer_summary_v2.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("✅ Done. Artifacts saved:")
    print(f"- Model: {MODELS_DIR / 'logreg_transformer_emb_best.joblib'}")
    print(f"- Label encoder: {MODELS_DIR / 'label_encoder_transformer.joblib'}")
    print(f"- Embeddings: {EMB_DIR}")
    print(f"- Reports: {REPORTS_DIR}")
    print(f"- Summary: {summary_path}")

if __name__ == "__main__":
    main()
