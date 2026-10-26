# Probabilistic SMS Spam Detection

BAMAT207 Probability and Statistics — Project Activity

**Student:** Kushal Bagla  
**Roll number:** 25BAI0044  
**Topic:** Probability and Random Variables  
**Submission deadline:** 27 October 2026

## Objective

Build a Python system that classifies SMS messages as spam or genuine messages (ham). Use dataset analysis, conditional probability, Bayes' theorem and a Bernoulli random variable to explain the data, then train and evaluate a Multinomial Naive Bayes classifier.

## Implementation plan

1. Set up the project and its Python dependencies.
2. Load the official UCI dataset, check missing values, remove duplicate messages and encode spam as 1 and ham as 0.
3. Summarize class counts and probabilities and create a class-distribution chart.
4. Calculate conditional probabilities for `free`, `prize`, `win`, `urgent` and `offer`, and verify them with Bayes' theorem.
5. Calculate the expectation and variance of the spam indicator random variable.
6. Split the data into 80% training and 20% testing with stratification; fit CountVectorizer only on training messages.
7. Train MultinomialNB and report accuracy, precision, recall, F1-score and a confusion matrix, including both types of classification error.
8. Accept a new SMS and display its predicted class and both model probabilities.

## Dataset and attribution

Almeida, T. and Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

UCI lists the dataset under the Creative Commons Attribution 4.0 International licence. Dataset attribution will be retained with the data. The source file's actual row count will be recorded during loading; the assignment quotes 5,574 messages.

## Current status

Project setup is in progress. Dataset processing, notebook results, model training and the final report will be added in subsequent stages. No evaluation results are claimed yet.

## Submission deliverables

- Executed Python notebook
- Cleaned dataset summary and class-distribution chart
- Probability calculations for all five specified words
- Trained classifier and its fitted vectorizer
- Confusion matrix and performance report
- Interactive prediction example
- GitHub repository link

Development is organized into meaningful local commits. Publishing to GitHub is handled by the repository owner.

## Local Python setup

Use Python 3.12. From the repository directory, run:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The listed direct dependency versions have been checked together using library imports and a small CountVectorizer and MultinomialNB smoke check. This check verifies basic library compatibility, not project accuracy. Transitive dependencies are not locked.

The notebook will be added in the dataset stage. Open it in a notebook-capable editor and select the `.venv` Python interpreter, or upload it to Google Colab. Colab setup instructions will be included with the notebook. The dataset will be downloaded from UCI using Python's standard library, so `ucimlrepo` is not required.
