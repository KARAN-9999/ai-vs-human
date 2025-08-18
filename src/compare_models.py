# src/compare_models.py

import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
PLOTS_DIR = REPORTS_DIR / "plots"
EMB_DIR = Path("data/embeddings")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

model_comparison = {}

# Helper: Safe ROC calculation
def safe_roc_auc(y_true, y_scores, label):
    if label not in y_true:
        return None  # Avoid undefined metric
    fpr, tpr, _ = roc_curve((y_true == label).astype(int), y_scores)
    return auc(fpr, tpr)

# Helper: Confusion Matrix Plot
def plot_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"confusion_matrix_{model_name}.png")
    plt.close()

# --- 1) Try TF-IDF Model ---
try:
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    tfidf_model = joblib.load(MODELS_DIR / "logreg_tfidf.joblib")

    # Load val set
    import pandas as pd
    val_df = pd.read_csv("data/processed/val_clean.csv")
    X_val_tfidf = vectorizer.transform(val_df["text"])
    y_val = val_df["label"].values

    y_pred_tfidf = tfidf_model.predict(X_val_tfidf)
    report_tfidf = classification_report(y_val, y_pred_tfidf, output_dict=True)
    model_comparison["TF-IDF_LogReg"] = report_tfidf["accuracy"]

    # Confusion Matrix
    cm_tfidf = confusion_matrix(y_val, y_pred_tfidf)
    plot_confusion_matrix(cm_tfidf, classes=np.unique(y_val), model_name="TF-IDF_LogReg")

except FileNotFoundError:
    print("⚠ TF-IDF model not found. Skipping...")
except Exception as e:
    print(f"⚠ TF-IDF evaluation failed: {e}")

# --- 2) SBERT Model ---
try:
    sbert_model = joblib.load(MODELS_DIR / "logreg_sbert.joblib")
    X_val = np.load(EMB_DIR / "val_embeddings.npy")
    y_val = np.load(EMB_DIR / "val_labels.npy")

    y_pred_sbert = sbert_model.predict(X_val)
    report_sbert = classification_report(y_val, y_pred_sbert, output_dict=True)
    model_comparison["SBERT_LogReg"] = report_sbert["accuracy"]

    # Confusion Matrix
    cm_sbert = confusion_matrix(y_val, y_pred_sbert)
    plot_confusion_matrix(cm_sbert, classes=np.unique(y_val), model_name="SBERT_LogReg")

except FileNotFoundError:
    print("⚠ SBERT model not found. Skipping...")
except Exception as e:
    print(f"⚠ SBERT evaluation failed: {e}")

# --- Save Reports ---
with open(REPORTS_DIR / "model_comparison.json", "w") as f:
    json.dump(model_comparison, f, indent=4)

with open(REPORTS_DIR / "model_comparison.md", "w") as f:
    f.write("# Model Comparison\n\n")
    for k, v in model_comparison.items():
        f.write(f"- {k}: {v:.4f}\n")

print("✅ Model comparison saved to model_comparison.json & model_comparison.md")
print(f"📊 Confusion matrices saved in {PLOTS_DIR}")
