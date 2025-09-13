import argparse, json
from pathlib import Path
import joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def main(data_path, out_dir):
    df = pd.read_csv(data_path).dropna()
    df = df[df["label"].isin(["AI","Human"])]
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    X_train, X_temp, y_train, y_temp = train_test_split(df["text"], y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    Xtr, Xv, Xte = vec.fit_transform(X_train), vec.transform(X_val), vec.transform(X_test)

    clf = LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=42)
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds, target_names=le.classes_, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)
    }

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, out/"tfidf.joblib")
    joblib.dump(clf, out/"model.joblib")
    joblib.dump(le, out/"labels.joblib")
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2))

    print("✅ Model trained")
    print("Accuracy:", metrics["accuracy"])
    print("Saved to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/dataset_clean.csv")
    ap.add_argument("--output_dir", default="models/lr_v1")
    args = ap.parse_args()
    main(args.data, args.output_dir)
