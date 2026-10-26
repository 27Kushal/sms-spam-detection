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
