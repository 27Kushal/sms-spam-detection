"""Empirical message-level probabilities for the cleaned SMS collection."""
import re

import numpy as np
import pandas as pd

SUSPICIOUS_WORDS = ("free", "prize", "win", "urgent", "offer")


def analyze_probabilities(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Count whole-word presence once per message, ignoring letter case.

    Python regex word boundaries exclude substrings such as 'win' in 'winning'.
    Undefined conditional probabilities are NaN, never a manufactured zero.
    """
    if clean.empty:
        raise ValueError("Probability analysis requires at least one message.")
    if clean[["message", "target"]].isna().any().any():
        raise ValueError("Messages and targets must not be missing.")
    if not clean["target"].isin([0, 1]).all():
        raise ValueError("Targets must be ham = 0 or spam = 1.")
    total = len(clean)
    spam = clean["target"].eq(1)
    n_spam = int(spam.sum())
    p_spam = n_spam / total
    rows = []
    for word in SUSPICIOUS_WORDS:
        contains = clean["message"].str.contains(r"\b" + re.escape(word) + r"\b", case=False, regex=True)
        n_word = int(contains.sum())
        n_joint = int((contains & spam).sum())
        p_word = n_word / total
        p_word_given_spam = n_joint / n_spam if n_spam else np.nan
        direct = n_joint / n_word if n_word else np.nan
        bayes = p_word_given_spam * p_spam / p_word if n_spam and n_word else np.nan
        rows.append({
            "word": word,
            "messages_with_word": n_word,
            "spam_with_word": n_joint,
            "ham_with_word": n_word - n_joint,
            "p_word": p_word,
            "p_word_given_spam": p_word_given_spam,
            "p_spam_given_word": direct,
            "bayes_p_spam_given_word": bayes,
        })
    summary = {
        "total_messages": total,
        "spam_messages": n_spam,
        "ham_messages": total - n_spam,
        "p_spam": p_spam,
        "p_ham": 1 - p_spam,
        "expectation_Y": p_spam,
        "variance_Y": p_spam * (1 - p_spam),
        "empirical_mean_Y": float(np.mean(clean["target"].to_numpy())),
        "empirical_population_variance_Y": float(np.var(clean["target"].to_numpy(), ddof=0)),
    }
    return pd.DataFrame(rows), summary
