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


Week 1 Summary – AI vs Human Text Classification Project
1. Dataset Overview
Source: Pre-collected dataset for AI vs Human text classification.

Total Samples: 7,000

Labels:

AI-generated: 3,500

Human-written: 3,500

Features:

text → The text sample.

generated → Confidence score (float).

label → Target variable (AI / Human).

length → Word count of the text.

2. Preprocessing Status
Dataset stored in data/processed and loaded into Jupyter Notebook.

Checked for missing values → No missing entries found.

Verified balanced class distribution.

Confirmed dataset is clean (no special characters or HTML tags).

3. Exploratory Data Analysis (EDA)
3.1 Class Distribution
AI and Human classes are perfectly balanced (3,500 each).

3.2 Text Length Statistics
Metric	Value
Min	100 words
Max	400 words
Mean	~290 words
Std Dev	~69.96 words
Short texts (<3 words)	0
Long texts (>100 words)	6,957

3.3 Word Clouds
AI Text Word Cloud: Frequent words include people, help, student, time, make.

Human Text Word Cloud: Frequent words include student, school, people, car, think.

4. Insights
The dataset is balanced, which is ideal for training.

Text lengths are consistent (between 100–400 words), making preprocessing simpler.

Some keywords overlap between AI and Human texts, meaning classification may rely on subtle linguistic differences rather than unique words.

Dataset quality is high → no major cleaning required.

5. Week 1 Deliverables
✅ Dataset structured in data/raw and data/processed.
✅ Jupyter Notebook 01_EDA.ipynb created and committed to GitHub.
✅ Class distribution, statistics, and visualizations completed.
✅ Initial insights documented.
✅ Repository pushed to GitHub: ai-vs-human

Next Steps (Week 2 Preview)
Implement text preprocessing pipeline (tokenization, stopword removal, lemmatization).

Convert text into numerical features using TF-IDF or embeddings.

Train baseline classification models.

Evaluate with metrics like accuracy, precision, recall, F1-score.