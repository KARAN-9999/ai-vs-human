# Transformer Embeddings - Summary (v2)
## Model & Setup
- Encoder: distilroberta-base
- LogisticRegression grid C: [0.01, 0.1, 1, 10]
- Selected best_C: 10
- Device used: cpu

## Label mapping
- 0 => AI
- 1 => Human

## Validation metrics
- accuracy: 0.9866
- precision: 0.9866
- recall: 0.9866
- f1: 0.9865
- roc_auc: 0.9984

## Test metrics
- accuracy: 0.9893
- precision: 0.9893
- recall: 0.9893
- f1: 0.9893
- roc_auc: 0.9995

- val ROC AUC: 0.9984
- test ROC AUC: 0.9995
