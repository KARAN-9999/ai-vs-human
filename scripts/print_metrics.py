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
