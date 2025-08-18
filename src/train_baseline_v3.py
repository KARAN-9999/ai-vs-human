import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from pathlib import Path

# Paths
DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Load cleaned data
train_df = pd.read_csv(DATA_DIR / "train_clean.csv")
val_df = pd.read_csv(DATA_DIR / "val_clean.csv")

X_train, y_train = train_df["text"], train_df["label"]
X_val, y_val = val_df["text"], val_df["label"]

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "SGDClassifier": SGDClassifier(loss='log_loss', max_iter=2000),
    "LinearSVC": LinearSVC()
}

results = {}
best_model_name, best_model, best_acc = None, None, 0

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_val_tfidf)
    
    acc = accuracy_score(y_val, y_pred)
    results[name] = {
        "accuracy": acc,
        "precision": precision_score(y_val, y_pred, pos_label="AI"),
        "recall": recall_score(y_val, y_pred, pos_label="AI"),
        "f1": f1_score(y_val, y_pred, pos_label="AI")
    }
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

# Save best model + vectorizer
joblib.dump(best_model, MODELS_DIR / f"{best_model_name.lower()}_tfidf.joblib")
joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")

# Save metrics
with open(REPORTS_DIR / "baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Best model: {best_model_name}")

# Error analysis
y_val_pred = best_model.predict(X_val_tfidf)
val_df["predicted"] = y_val_pred
false_pos = val_df[(val_df["label"] == "Human") & (val_df["predicted"] == "AI")]
false_neg = val_df[(val_df["label"] == "AI") & (val_df["predicted"] == "Human")]

error_samples = pd.concat([
    false_pos.assign(error_type="False Positive"),
    false_neg.assign(error_type="False Negative")
])
error_samples.to_csv(REPORTS_DIR / "error_samples.csv", index=False)

# Top features (only for interpretable models)
if best_model_name in ["LogisticRegression", "SGDClassifier"]:
    feature_names = vectorizer.get_feature_names_out()
    coefs = best_model.coef_[0]
    top_pos = sorted(zip(coefs, feature_names), reverse=True)[:20]
    top_neg = sorted(zip(coefs, feature_names))[:20]
    top_features_df = pd.DataFrame({
        "AI": [f for _, f in top_pos],
        "Human": [f for _, f in top_neg]
    })
    top_features_df.to_csv(REPORTS_DIR / "top_features.csv", index=False)

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred, labels=["AI", "Human"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["AI", "Human"], yticklabels=["AI", "Human"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig(REPORTS_DIR / "confusion_matrix.png")
plt.close()

# ROC curve (only if model has decision_function or predict_proba)
try:
    if hasattr(best_model, "decision_function"):
        y_scores = best_model.decision_function(X_val_tfidf)
    else:
        y_scores = best_model.predict_proba(X_val_tfidf)[:, 1]
    fpr, tpr, _ = roc_curve(y_val.map({"Human": 0, "AI": 1}), y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{best_model_name} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(REPORTS_DIR / "roc_curve.png")
    plt.close()
except Exception as e:
    print(f"ROC curve skipped: {e}")
