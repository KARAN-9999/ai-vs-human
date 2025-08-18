# src/train_baseline_v4.py

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
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

# Models to try
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=2000),
    "LinearSVC": LinearSVC()
}

results = {}
best_model_name = None
best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_val_tfidf)
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="weighted", zero_division=0)
    rec = recall_score(y_val, preds, average="weighted", zero_division=0)
    f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
    
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    if acc > best_acc:
        best_acc = acc
        best_model_name = name
        best_model = model

# Save vectorizer and best model
joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
joblib.dump(best_model, MODELS_DIR / f"{best_model_name.lower()}_tfidf.joblib")

# Save metrics
with open(REPORTS_DIR / "baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)

print(f"Best model: {best_model_name}")

# Extract top features
feature_names = np.array(vectorizer.get_feature_names_out())
try:
    if hasattr(best_model, "coef_"):
        coefs = best_model.coef_[0]
    elif hasattr(best_model, "dual_coef_"):
        coefs = best_model.dual_coef_[0]
    else:
        raise AttributeError("No coefficients found in the model.")
except Exception as e:
    print("⚠ Could not extract coefficients:", e)
    coefs = None

if coefs is not None:
    top_n = 20
    top_positive_indices = np.argsort(coefs)[-top_n:]
    top_negative_indices = np.argsort(coefs)[:top_n]
    
    top_features = pd.DataFrame({
        "feature": np.concatenate([feature_names[top_positive_indices], feature_names[top_negative_indices]]),
        "weight": np.concatenate([coefs[top_positive_indices], coefs[top_negative_indices]]),
        "class": ["AI"] * top_n + ["Human"] * top_n
    })
    
    top_features.to_csv(REPORTS_DIR / "top_features.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ["red" if c == "AI" else "blue" for c in top_features["class"]]
    plt.barh(top_features["feature"], top_features["weight"], color=colors)
    plt.title(f"Top Features for {best_model_name}")
    plt.xlabel("Coefficient Weight")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "top_features.png")
    plt.close()
    print("✅ Saved top_features.csv and top_features.png")
else:
    print("⚠ Skipped top features extraction — model has no coefficients.")

print("All outputs saved in 'models/' and 'reports/' folders.")
