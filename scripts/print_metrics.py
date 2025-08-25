import json

files = [
    ("SBERT", "reports/sbert_results.json"),
    ("Transformer v2", "reports/transformer_emb_results_v2.json")
]

for name, path in files:
    with open(path) as f:
        data = json.load(f)
    test_metrics = data.get("test", data)  # fallback if no "test" key
    print(f"\n{name} Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np
import os
import pandas as pd
os.makedirs("reports", exist_ok=True)

# Example: plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"reports/{filename}", bbox_inches="tight")
    plt.close()

# Example: ROC curve
def plot_roc(y_true, y_proba, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(f"reports/{filename}", bbox_inches="tight")
    plt.close()
df = pd.read_csv("reports/transformer_errors_test.csv")
y_test = df["true_label"]
y_pred = df["predicted_label"]


plot_confusion_matrix(y_test, y_pred, "Transformer Confusion Matrix (Test)", "cm_transformer_test.png")



