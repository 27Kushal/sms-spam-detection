# Submission guide

Student: Kushal Bagla  
Roll number: 25BAI0044  
Course: BAMAT207 Probability and Statistics  
Deadline: 27 October 2026

## Required deliverables

| Assignment requirement | Completed file |
| --- | --- |
| Python or Google Colab notebook | `sms_spam_detection.ipynb` |
| Cleaned dataset summary | `reports/dataset_summary.json` and notebook Activity 1 |
| Cleaned messages | `data/processed/sms_cleaned.csv` |
| Class-distribution chart | `reports/figures/class_distribution.png` |
| Five suspicious-word probability calculations | `reports/word_probabilities.csv`, `reports/probability_analysis.md` and notebook Activity 3 |
| Expectation and variance | `reports/probability_summary.json` and notebook Activity 4 |
| Stratified 80/20 split | `reports/split_manifest.csv` and notebook Activity 5 |
| Trained Naive Bayes model and vectorizer | `models/sms_naive_bayes.joblib` |
| Confusion matrix | `reports/figures/confusion_matrix.png` |
| Performance report | `reports/classification_report.txt`, `reports/model_metrics.json` and `reports/model_evaluation.md` |
| Both types of error | `reports/misclassified_messages.csv` |
| Real-time prediction | `python -m src.predict` and notebook Activity 8 |
| Formatted project report | `reports/Project_Report.docx` |
| GitHub submission link | https://github.com/27Kushal/sms-spam-detection |

## Run and verify

Use Python 3.12 and install the pinned dependencies following README.md. Run the notebook from the repository root to regenerate the data and model outputs. Set RUN_INTERACTIVE = True only when you want to enter messages. Run `python -m unittest discover -s tests -v` for the 17 unit tests.

The notebook is self-contained for upload to Google Colab and can download the official dataset if absent. For matching library versions, upload requirements.txt and install it before running the notebook. The checked local environment is recorded in models/training_metadata.json; results can vary with different library versions. A saved joblib model should only be loaded from a trusted source.

## Publish and submit

The final work is committed locally. From the repository directory, run:

```bash
git push
```

Then open the repository on GitHub and check that the latest notebook and reports/Project_Report.docx are present. Ensure your instructor can access the repository, and submit the repository link with the notebook/report as required by the course. No code was pushed automatically.

Repository link: https://github.com/27Kushal/sms-spam-detection
