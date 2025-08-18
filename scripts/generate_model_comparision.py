import json
import pandas as pd
from pathlib import Path

# Paths
reports_dir = Path("reports")
sbert_path = reports_dir / "sbert_results.json"
transformer_path = reports_dir / "transformer_emb_results_v2.json"
sbert_errors_path = reports_dir / "sbert_errors_test.csv"
transformer_errors_path = reports_dir / "transformer_errors_test.csv"

# Read metrics
with open(sbert_path) as f:
    sbert_metrics = json.load(f)
sbert_test = sbert_metrics.get("test", sbert_metrics)

with open(transformer_path) as f:
    transformer_metrics = json.load(f)
transformer_test = transformer_metrics.get("test", transformer_metrics)

# Compute improvements
f1_improvement = transformer_test["f1"] - sbert_test["f1"]

# JSON output
comparison_json = {
    "SBERT_test": sbert_test,
    "Transformer_v2_test": transformer_test,
    "F1_improvement": f1_improvement
}

# Save JSON
json_out_path = reports_dir / "model_comparison.json"
with open(json_out_path, "w") as f:
    json.dump(comparison_json, f, indent=4)

# Read top-5 errors
def get_top_errors(path, n=5):
    if path.exists():
        df = pd.read_csv(path)
        return df.head(n)
    return pd.DataFrame()

sbert_errors = get_top_errors(sbert_errors_path)
transformer_errors = get_top_errors(transformer_errors_path)

# Markdown output
comparison_md = f"""# Model Comparison – SBERT vs Transformer v2

**Test Set Performance**

| Metric     | SBERT  | Transformer v2 |
|------------|--------|----------------|
| Accuracy   | {sbert_test['accuracy']:.4f} | {transformer_test['accuracy']:.4f} |
| Precision  | {sbert_test['precision']:.4f} | {transformer_test['precision']:.4f} |
| Recall     | {sbert_test['recall']:.4f} | {transformer_test['recall']:.4f} |
| F1 Score   | {sbert_test['f1']:.4f} | {transformer_test['f1']:.4f} |
| ROC AUC    | N/A    | {transformer_test.get('roc_auc', 0):.4f} |

**F1 Improvement:** `{f1_improvement:.4f}`

---

### Visuals

#### Transformer v2 (Test)
![Transformer Confusion Matrix](cm_transformer_test.png)
![Transformer ROC Curve](roc_transformer_test.png)

#### SBERT (Test)
*(Add confusion matrix if available)*

---

## Top-5 Misclassified Examples – Transformer v2
"""
if not transformer_errors.empty:
    comparison_md += transformer_errors.to_markdown(index=False)
else:
    comparison_md += "_No error samples found._"

comparison_md += "\n\n## Top-5 Misclassified Examples – SBERT\n"
if not sbert_errors.empty:
    comparison_md += sbert_errors.to_markdown(index=False)
else:
    comparison_md += "_No error samples found._"

# Save Markdown
md_out_path = reports_dir / "model_comparison.md"
with open(md_out_path, "w", encoding="utf-8") as f:
    f.write(comparison_md)

print(f"✅ Comparison files saved:\n - {json_out_path}\n - {md_out_path}")
