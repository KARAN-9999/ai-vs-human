import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import seaborn as sns

# Load data
train = pd.read_csv("data/processed/train_clean.csv")
val = pd.read_csv("data/processed/val_clean.csv")

# Vectorize
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=5, max_df=0.9)
X_train = vectorizer.fit_transform(train['text'])
X_val = vectorizer.transform(val['text'])

y_train = train['label']
y_val = val['label']

# Models to try
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "SGDClassifier": SGDClassifier(loss='log_loss', max_iter=2000),
    "LinearSVC": LinearSVC(max_iter=2000)
}

results = {}
best_acc = 0
best_model = None
best_name = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, preds, average='weighted')
    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# Save best model + vectorizer
joblib.dump(best_model, f"models/{best_name}_best.joblib")
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

# Save results JSON
with open("reports/baseline_results.json", "w") as f:
    json.dump(results, f, indent=4)

# Confusion matrix for best model
preds = best_model.predict(X_val)
cm = confusion_matrix(y_val, preds)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"{best_name} Confusion Matrix")
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig("reports/plots/confusion_matrix.png")
plt.close()

print("Best model:", best_name)
print("Results saved to reports/")
