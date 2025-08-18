import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Paths
ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

def load_data():
    try:
        train_df = pd.read_csv(PROCESSED / "train_clean.csv")
        val_df   = pd.read_csv(PROCESSED / "val_clean.csv")
        test_df  = pd.read_csv(PROCESSED / "test_clean.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing cleaned CSV files. Run preprocess.py first. {e}")
    
    # Handle any NaN text
    for df in [train_df, val_df, test_df]:
        df['text'] = df['text'].fillna("").astype(str)
        df['label'] = df['label'].fillna("Unknown").astype(str)
    
    return train_df, val_df, test_df

def train_tfidf_logreg(train_df, val_df):
    # Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )

    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']

    X_val = vectorizer.transform(val_df['text'])
    y_val = val_df['label']

    # Logistic Regression Model
    model = LogisticRegression(
        C=2,
        max_iter=1000,
        class_weight='balanced',
        solver='liblinear'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # Save model & vectorizer
    joblib.dump(model, MODELS / "logreg_tfidf.pkl")
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.pkl")
    print(f"Model and vectorizer saved in {MODELS}")

    # Save report
    with open(REPORTS / "week2_baseline_results.md", "w") as f:
        f.write(f"# Baseline Model Results\n\n")
        f.write(f"**Validation Accuracy**: {acc:.4f}\n\n")
        f.write("## Classification Report\n")
        f.write(classification_report(y_val, y_pred))
        f.write("\n\n## Confusion Matrix\n")
        f.write(np.array2string(confusion_matrix(y_val, y_pred)))

if __name__ == "__main__":
    train_df, val_df, test_df = load_data()
    train_tfidf_logreg(train_df, val_df)
