"""Regression checks for parsing and cleaning decisions that affect results."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import clean_dataset, download_dataset, load_dataset


class DataTests(unittest.TestCase):
    def test_quotes_tabs_and_unicode_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messages.txt"
            path.write_text('ham\t"Meet\tfor café?"\nspam\tFree prize\n', encoding="utf-8")
            frame = load_dataset(path)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.loc[0, "message"], '"Meet\tfor café?"')

    def test_missing_blank_duplicates_and_encoding(self):
        raw = pd.DataFrame({
            "source_line": range(1, 7),
            "label": ["ham", "ham", "spam", "ham", None, "spam"],
            "message": ["Hello", "Hello", "Free prize", "  ", "Hi", None],
        })
        clean, summary = clean_dataset(raw)
        self.assertEqual(clean["target"].tolist(), [0, 1])
        self.assertEqual(summary["removed_missing_or_blank_records"], 3)
        self.assertEqual(summary["removed_duplicate_messages"], 1)
        self.assertEqual(clean["source_line"].tolist(), [1, 3])

    def test_conflicting_labels_are_rejected(self):
        raw = pd.DataFrame({"source_line": [1, 2], "label": ["ham", "spam"], "message": ["same", "same"]})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            clean_dataset(raw)

    def test_modified_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "messages.txt"
            path.write_text("ham\taltered data\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                download_dataset(path)

    def test_official_source_integrity_and_cleaning(self):
        path = Path(__file__).resolve().parents[1] / "data/raw/SMSSpamCollection"
        raw = load_dataset(download_dataset(path))
        clean, summary = clean_dataset(raw)
        self.assertEqual(len(raw), 5574)
        self.assertFalse(clean["message"].duplicated().any())
        self.assertEqual(len(raw), len(clean) + summary["removed_missing_or_blank_records"] + summary["removed_duplicate_messages"])
        self.assertTrue((clean["target"] == clean["label"].map({"ham": 0, "spam": 1})).all())


if __name__ == "__main__":
    unittest.main()
