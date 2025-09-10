"""
Robust Baseline Training Script (train_baseline_v5.py)
Improves model generalization using data augmentation, cross-validation,
and a regularized TF-IDF + Logistic Regression pipeline.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# Paths
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation data."""

    train_full = DATA_DIR / "train_full_clean.csv"
    val_full = DATA_DIR / "val_full_clean.csv"
    if train_full.exists() and val_full.exists():
        print("[data] Using combined dataset for training.")
        train_df = pd.read_csv(train_full)
        val_df = pd.read_csv(val_full)
        return train_df, val_df

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


def create_robust_pipeline() -> Pipeline:
    """Create a more generalizable model pipeline."""

    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,         # Reduced features to avoid overfitting
            ngram_range=(1, 2),         # Unigrams and bigrams
            min_df=3,                   # Ignore rare terms
            max_df=0.8,                 # Ignore very common terms
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            C=0.1,                     # Strong regularization
            class_weight='balanced',   # Handle class imbalance
            max_iter=1000,
            random_state=42
        ))
    ])


def data_augmentation_techniques(df: pd.DataFrame) -> pd.DataFrame:
    """Add data augmentation to increase diversity."""

    import re

    augmented_data = []

    for _, row in df.iterrows():
        text = row['text']
        label = row['label']

        # Original text
        augmented_data.append({'text': text, 'label': label})

        # Lowercase variation
        augmented_data.append({'text': text.lower(), 'label': label})

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text).strip()
        augmented_data.append({'text': cleaned, 'label': label})

        # Random word order shuffle (for AI-generated text only)
        if label == 'AI':
            words = text.split()
            if len(words) > 10:
                np.random.shuffle(words)
                shuffled = ' '.join(words)
                augmented_data.append({'text': shuffled, 'label': label})

    return pd.DataFrame(augmented_data)


def cross_validate_model(X: pd.Series, y: pd.Series, cv_folds: int = 5) -> float:
    """Proper cross-validation to detect overfitting."""

    pipeline = create_robust_pipeline()
    kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    scores = []

    for train_idx, val_idx in kfold.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_val, y_val)
        scores.append(score)

    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {np.mean(scores):.4f} (+/- {np.std(scores)*2:.4f})")
    return np.mean(scores)


def save_classification_report(y_true, y_pred, filename: Path):
    report = classification_report(y_true, y_pred, zero_division=0)
    with open(filename, "w") as f:
        f.write(report)
    print(f"[report] Classification report saved to {filename}")


def save_confusion_matrix(y_true, y_pred, filename: Path):
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(filename, cm, fmt='%d', delimiter=',')
    print(f"[report] Confusion matrix saved to {filename}")


def main() -> None:
    # Load data
    train_df, val_df = load_dataset()

    # Combine train + val sets for cross-validation and final training
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    X_full = full_df["text"].astype(str)
    y_full = full_df["label"].astype(str)

    print("[info] Augmenting data...")
    augmented_df = data_augmentation_techniques(full_df)
    X_aug = augmented_df["text"].astype(str)
    y_aug = augmented_df["label"].astype(str)

    print("[info] Cross-validating model...")
    mean_cv_score = cross_validate_model(X_aug, y_aug, cv_folds=5)
    print(f"[info] Mean cross-validation accuracy: {mean_cv_score:.4f}")

    # Train final model on entire augmented dataset
    print("[info] Training final model on full augmented dataset...")
    pipeline = create_robust_pipeline()
    pipeline.fit(X_aug, y_aug)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "robust_baseline_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[model] Final model saved at {model_path}")

    # Evaluate on original validation set
    X_val = val_df["text"].astype(str)
    y_val = val_df["label"].astype(str)
    y_pred = pipeline.predict(X_val)

    # Save classification report and confusion matrix on val set
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_classification_report(y_val, y_pred, REPORTS_DIR / "classification_report_val.txt")
    save_confusion_matrix(y_val, y_pred, REPORTS_DIR / "confusion_matrix_val.csv")

    # Optionally, plot top features (only applicable for LogisticRegression)
    try:
        if hasattr(pipeline.named_steps['classifier'], "coef_"):
            feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
            coefs = pipeline.named_steps['classifier'].coef_[0]
            top_n = 20
            top_positive_indices = np.argsort(coefs)[-top_n:]
            top_negative_indices = np.argsort(coefs)[:top_n]
            top_features = pd.DataFrame({
                "feature": np.concatenate([feature_names[top_positive_indices], feature_names[top_negative_indices]]),
                "weight": np.concatenate([coefs[top_positive_indices], coefs[top_negative_indices]]),
                "class": ["AI"] * top_n + ["Human"] * top_n
            })
            plt.figure(figsize=(10, 6))
            colors = ["red" if c == "AI" else "blue" for c in top_features["class"]]
            plt.barh(top_features["feature"], top_features["weight"], color=colors)
            plt.title("Top Features - Robust Baseline Model")
            plt.xlabel("Coefficient Weight")
            plt.tight_layout()
            plt.savefig(REPORTS_DIR / "top_features.png")
            plt.close()
            top_features.to_csv(REPORTS_DIR / "top_features.csv", index=False)
            print("[report] Saved top_features.csv and top_features.png")
    except Exception as e:
        print(f"[warn] Could not extract or plot top features: {e}")

    print("[info] Training pipeline finished. All reports and model saved.")


if __name__ == "__main__":
    main()
