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

Stages 1 to 3 are complete: project setup, dataset handling and probability analysis. The executed notebook covers assignment Activities 1 to 4, including the class-distribution chart, all five word probabilities, Bayes checks and Bernoulli expectation and variance. Model training, evaluation and the final report remain for later stages. No model evaluation results are claimed yet.

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

Open `sms_spam_detection.ipynb` in a notebook-capable editor from the repository root and select the `.venv` Python interpreter. Run all cells in order. Alternatively, upload the notebook to Google Colab; its data functions are self-contained and it downloads the official source if absent. Pandas, NumPy and Matplotlib are required and are normally preinstalled in Colab. The download uses Python's standard library, so `ucimlrepo` is not required.


## Dataset handling results

| Class | Original records | Duplicate messages removed | Cleaned records |
| --- | ---: | ---: | ---: |
| Ham | 4,827 | 309 | 4,518 |
| Spam | 747 | 94 | 653 |
| Total | 5,574 | 403 | 5,171 |

There are no missing or blank labels/messages. Deduplication retains the first occurrence of each exact message text. Case, punctuation and whitespace in nonblank messages are preserved. Conflicting labels for the same message would stop processing rather than silently choose a label; none were found. Ham is encoded as 0 and spam as 1.

The parser splits each source line at the first tab and treats quotation marks as literal SMS content. The raw file is checked against a recorded SHA-256 digest before use. Duplicate removal precedes the future training/test split.

## Files completed so far

- `sms_spam_detection.ipynb`: executed notebook with explanations and cleaning outputs.
- `src/data.py`: reusable downloading, loading and cleaning functions.
- `data/raw/`: unchanged official dataset and its original documentation.
- `data/processed/sms_cleaned.csv`: cleaned messages, numeric labels and original source line numbers.
- `reports/dataset_summary.json`: machine-readable cleaning summary.
- `tests/test_data.py`: parser, validation and cleaning regression checks.

## Verification

From the repository root, with the project environment activated:

```bash
python -m unittest discover -s tests -v
```

Five data checks cover literal quotation marks and embedded tabs, missing/blank records, deduplication and encoding, conflicting labels, checksum validation and official dataset integrity. The executed notebook additionally confirms that saving and reloading the cleaned CSV preserves the data exactly.


## Probability analysis results

Spam accounts for 12.6281% of the cleaned messages and ham for 87.3719%. The Bernoulli spam indicator has expectation 0.12628118 and variance 0.11033425.

| Word | P(Spam given word) |
| --- | ---: |
| free | 72.9064% |
| prize | 100.0000% |
| win | 89.0909% |
| urgent | 91.9355% |
| offer | 80.0000% |

These are empirical conditional probabilities for case-insensitive whole-word presence, counted once per message. They do not guarantee outcomes for new SMS messages. All five direct calculations agree with Bayes' theorem.

Additional outputs:

- `reports/figures/class_distribution.png`: labelled class-count chart.
- `reports/word_probabilities.csv`: message counts and conditional probabilities.
- `reports/probability_summary.json`: class probabilities and Bernoulli moments.
- `reports/probability_analysis.md`: formulas, results and interpretation.
- `src/probability.py`: reusable probability-analysis functions.
- `tests/test_probability.py`: four checks covering word boundaries, repeated words, Bayes calculations, moments, absent words, a missing spam class and empty input.

All nine unit tests passed. The notebook cells executed successfully, all five Bayes equalities passed, and the class-distribution chart was visually checked.
