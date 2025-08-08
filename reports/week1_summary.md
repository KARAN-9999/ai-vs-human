Capstone Project – Week 1 Summary
Project Title: AI vs Human Text Classification
Student Name: Karan
Date: 08-08-2025

1. Dataset Source
Dataset Name: AI vs Human Text 
DataSet Link : https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

Reason for Selection:

Large dataset (~X rows) with both AI and Human text.

Well-labeled (generated → AI / Human).

Diverse topics → improves model generalization.

2. Dataset Preprocessing
Steps Taken:

Unzipped and placed dataset in data/raw/.

Loaded in chunks to handle large file size.

Cleaned text:

Removed HTML tags, URLs, AI disclaimers.

Normalized whitespace and Unicode characters.

Filtered texts between 100–400 words for fairness.

Deduplicated using hash checks.

Balanced dataset → equal AI and Human samples (X each).

Saved cleaned dataset to data/processed/.

Created train/val/test splits (70/15/15).

3. EDA Findings
Class Balance:

AI: X samples

Human: X samples

Text Length:

Median length ~ Y words for both classes.

Distributions overlap → no obvious length bias.

Qualitative Observations:

AI texts often contain structured explanations.

Human texts show more stylistic variety.

4. Challenges & Solutions
Challenge	Solution
Large CSV size	Processed in chunks with Pandas
Noisy text	Regex cleaning + HTML unescape
Class imbalance	Downsampled majority class

5. Next Steps (Week-2 Plan)
Start feature extraction with TF-IDF baseline.

Train initial Logistic Regression / SVM model for benchmark.

Evaluate accuracy, precision, recall, F1-score.

Document baseline results.

Attachments:

dataset_cleaned_sample.csv

EDA plots (class distribution, text length)