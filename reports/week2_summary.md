Week 2 Model Performance Summary
Comparison of SBERT vs Transformer v2
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
SBERT	0.9266	0.9267	0.9266	0.9266	–
Transformer v2	0.9866	0.9867	0.9866	0.9866	0.9992

Performance Insights
Transformer v2 significantly outperforms SBERT across all metrics.

Gains of ~6% absolute in accuracy and F1.

Near-perfect ROC-AUC of 0.9992, indicating excellent separability between AI and Human classes.

High precision and recall balance means it’s reliable for both detecting AI-generated and human-written text.

Week 2 – Transformer (v2) Embedding + Logistic Regression Results
Model:
Transformer backbone: distilroberta-base (HuggingFace)

Classifier: Logistic Regression (Grid Search over C = [0.01, 0.1, 1, 10])

Best hyperparameter: C = 10 (highest validation F1)

1. Validation Set Performance
Metric	Score
Accuracy	0.98597
Precision	0.98598
Recall	0.98597
F1-Score	0.98597
ROC-AUC	0.9991

Plots:

Confusion Matrix – reports/cm_transformer_val.png

ROC Curve – reports/roc_transformer_val.png

2. Test Set Performance
Metric	Score
Accuracy	0.98664
Precision	0.98665
Recall	0.98664
F1-Score	0.98664
ROC-AUC	0.9992

Plots:

Confusion Matrix – reports/cm_transformer_test.png

ROC Curve – reports/roc_transformer_test.png

3. Error Analysis
Validation errors: See reports/transformer_errors_val_top50.csv

Test errors: See reports/transformer_errors_test_top50.csv

Errors are rare (~1.4% on test set).

Most misclassifications occur on short, ambiguous sentences without clear stylistic cues.

4. Overfitting/Underfitting Analysis
Validation F1: 0.98597

Test F1: 0.98664

The scores are nearly identical → no significant overfitting or underfitting.

High ROC-AUC (> 0.999) shows strong separation between AI and Human text.

Model generalizes extremely well to unseen data.

5. Conclusion
Transformer embeddings + tuned Logistic Regression significantly outperform the Week 1 SBERT baseline (+6% absolute F1 gain).

The model is robust, with minimal performance drop from validation to test.

Ready for deployment in the classification pipeline.

Week 2 – Transformer (v2) Embedding + Logistic Regression Results
Model Setup
Transformer backbone: distilroberta-base (HuggingFace)

Classifier: Logistic Regression (Grid Search over C = [0.01, 0.1, 1, 10])

Best hyperparameter: C = 10 (highest validation F1)

1. Validation Set Performance
Metric	Score
Accuracy	0.98597
Precision	0.98598
Recall	0.98597
F1-Score	0.98597
ROC-AUC	0.9991

Confusion Matrix:


ROC Curve:


2. Test Set Performance
Metric	Score
Accuracy	0.98664
Precision	0.98665
Recall	0.98664
F1-Score	0.98664
ROC-AUC	0.9992

Confusion Matrix:


ROC Curve:


3. Error Analysis
Validation errors: See reports/transformer_errors_val_top50.csv

Test errors: See reports/transformer_errors_test_top50.csv

Errors are rare (~1.4% on test set).

Most misclassifications occur on short, ambiguous sentences without clear stylistic cues.

4. Overfitting/Underfitting Analysis
Validation F1: 0.98597

Test F1: 0.98664

Nearly identical scores → no significant overfitting or underfitting.

High ROC-AUC (> 0.999) shows strong separation between AI and Human text.

Model generalizes extremely well to unseen data.

5. Conclusion
Transformer embeddings + tuned Logistic Regression outperform Week 1 SBERT baseline (+6% absolute F1 gain).

The model is robust, with minimal performance drop from validation to test.

Ready for deployment in the classification pipeline.