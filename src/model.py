"""Reproducible data preparation, training and evaluation for SMS detection."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def split_messages(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve 20% for testing with class stratification and a fixed seed."""
    if clean["message"].duplicated().any():
        raise ValueError("Remove duplicate messages before splitting.")
    train, test = train_test_split(
        clean, test_size=0.20, random_state=RANDOM_STATE, stratify=clean["target"]
    )
    train, test = train.copy(), test.copy()
    if set(train["source_line"]) & set(test["source_line"]):
        raise ValueError("Source records overlap between training and testing.")
    return train, test


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_model(train: pd.DataFrame) -> Pipeline:
    """Fit word counts and MultinomialNB using training messages only."""
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])
    model.fit(train["message"], train["target"])
    return model


def evaluate_model(model: Pipeline, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Evaluate untouched test records; spam is the positive class."""
    predictions = model.predict(test["message"])
    probabilities = model.predict_proba(test["message"])
    class_columns = {int(label): i for i, label in enumerate(model.classes_)}
    matrix = confusion_matrix(test["target"], predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    report = classification_report(
        test["target"], predictions, labels=[0, 1], target_names=["ham", "spam"],
        output_dict=True, zero_division=0,
    )
    metrics = {
        "test_messages": len(test),
        "accuracy": float(accuracy_score(test["target"], predictions)),
        "spam_precision": float(report["spam"]["precision"]),
        "spam_recall": float(report["spam"]["recall"]),
        "spam_f1": float(report["spam"]["f1-score"]),
        "always_ham_baseline_accuracy": float(test["target"].eq(0).mean()),
        "confusion_matrix": matrix.tolist(),
        "matrix_label_order": ["ham", "spam"],
        "matrix_orientation": "rows = actual; columns = predicted",
        "true_negatives": tn, "false_positives": fp,
        "false_negatives": fn, "true_positives": tp,
        "classification_report": report,
    }
    results = test[["source_line", "label", "message", "target"]].copy()
    results["predicted_target"] = predictions
    results["predicted_label"] = np.where(predictions == 1, "spam", "ham")
    results["p_ham"] = probabilities[:, class_columns[0]]
    results["p_spam"] = probabilities[:, class_columns[1]]
    results["error_type"] = np.select(
        [(results["target"] == 1) & (predictions == 0),
         (results["target"] == 0) & (predictions == 1)],
        ["false_negative", "false_positive"], default="correct",
    )
    return metrics, results
