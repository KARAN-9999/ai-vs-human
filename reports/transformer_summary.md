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
- accuracy: 0.9872
- precision: 0.9873
- recall: 0.9872
- f1: 0.9872
- roc_auc: 0.9985

## Test metrics
- accuracy: 0.9906
- precision: 0.9906
- recall: 0.9906
- f1: 0.9906
- roc_auc: 0.9995

- val ROC AUC: 0.9985
- test ROC AUC: 0.9995
