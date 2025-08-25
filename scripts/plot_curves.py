import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# Load test errors file (has y_true, y_pred, y_proba)
df = pd.read_csv("reports/transformer_errors_test.csv")

y_true = df["true_label"]
y_proba = df["pred_proba"]

# Create reports folder if not exists
os.makedirs("reports", exist_ok=True)

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("Transformer ROC Curve (Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("reports/roc_transformer_test.png", bbox_inches="tight")
plt.close()

print("✅ ROC curve saved: reports/roc_transformer_test.png")
