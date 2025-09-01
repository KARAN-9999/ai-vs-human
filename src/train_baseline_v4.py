"""
Extended baseline training script (v4) for AI vs Human classification.

This version of the baseline training script automatically detects whether
combined dataset splits created by ``scripts/preprocess_all.py`` are
available. If ``data/processed/train_full_clean.csv`` exists the
script uses the ``*_full_clean.csv`` files; otherwise it falls back to
the original Kaggle splits (``train_clean.csv``, ``val_clean.csv``).

The script trains three linear classifiers (LogisticRegression,
SGDClassifier and LinearSVC) on TF‑IDF features, evaluates them on the
validation set and picks the best model based on accuracy. It also
extracts top features and saves metrics and plots to the ``reports``
directory. Models and vectorizer are saved to ``models``.

Run this script after you have cleaned your data using ``src/preprocess.py``.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Optional  # Ensure Optional is available for type hints
import matplotlib.pyplot as plt
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


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation data.

    Prefers the combined (``*_full_clean.csv``) splits if they exist. If not
    found, falls back to the Kaggle splits (``*_clean.csv``). Raises
    ``FileNotFoundError`` if none of the required files are present.

    Returns:
        A tuple (train_df, val_df).
    """
    # Attempt to load combined dataset
    train_full = DATA_DIR / "train_full_clean.csv"
    val_full = DATA_DIR / "val_full_clean.csv"
    if train_full.exists() and val_full.exists():
        print("[data] Using combined dataset for training.")
        train_df = pd.read_csv(train_full)
        val_df = pd.read_csv(val_full)
        return train_df, val_df
    # Fallback to Kaggle dataset
    train_clean = DATA_DIR / "train_clean.csv"
    val_clean = DATA_DIR / "val_clean.csv"
    if train_clean.exists() and val_clean.exists():
        print("[data] Using Kaggle dataset for training.")
        train_df = pd.read_csv(train_clean)
        val_df = pd.read_csv(val_clean)
        return train_df, val_df
    raise FileNotFoundError(
        "Could not find any cleaned datasets. Run src/preprocess.py first."
    )


def main() -> None:
    # Load data
    train_df, val_df = load_dataset()
    X_train = train_df["text"].astype(str)
    y_train = train_df["label"].astype(str)
    X_val = val_df["text"].astype(str)
    y_val = val_df["label"].astype(str)

    # TF‑IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Models to try
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2_000),
        "SGDClassifier": SGDClassifier(loss="log_loss", max_iter=2_000),
        "LinearSVC": LinearSVC()
    }
    results = {}
    best_model_name = None
    best_model = None
    best_acc = 0.0

    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_val_tfidf)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average="weighted", zero_division=0)
        rec = recall_score(y_val, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model

    # Persist vectorizer and best model
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best_model, MODELS_DIR / f"{best_model_name.lower()}_tfidf.joblib")

    # Save metrics
    with open(REPORTS_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"[model] Best model: {best_model_name}")

    # Extract top features (only for models with coefficients)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs: Optional[np.ndarray] = None
    try:
        if hasattr(best_model, "coef_"):
            coefs = best_model.coef_[0]
        elif hasattr(best_model, "dual_coef_"):
            coefs = best_model.dual_coef_[0]
    except Exception as ex:
        print("[warn] Could not extract coefficients from the best model:", ex)

    if coefs is not None:
        top_n = 20
        # top positive and negative coefficients
        top_positive_indices = np.argsort(coefs)[-top_n:]
        top_negative_indices = np.argsort(coefs)[:top_n]
        top_features = pd.DataFrame({
            "feature": np.concatenate([feature_names[top_positive_indices], feature_names[top_negative_indices]]),
            "weight": np.concatenate([coefs[top_positive_indices], coefs[top_negative_indices]]),
            "class": ["AI"] * top_n + ["Human"] * top_n
        })
        top_features.to_csv(REPORTS_DIR / "top_features.csv", index=False)
        # Plot top features
        plt.figure(figsize=(10, 6))
        colors = ["red" if c == "AI" else "blue" for c in top_features["class"]]
        plt.barh(top_features["feature"], top_features["weight"], color=colors)
        plt.title(f"Top Features for {best_model_name}")
        plt.xlabel("Coefficient Weight")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "top_features.png")
        plt.close()
        print("[report] Saved top_features.csv and top_features.png")
    else:
        print("[info] Skipped top feature extraction – best model has no coefficients.")

    print("All outputs saved in 'models/' and 'reports/' directories.")


if __name__ == "__main__":
    main()