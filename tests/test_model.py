"""Checks for disjoint, stratified data preparation and model evaluation."""
import unittest
import pandas as pd
from src.model import split_messages


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.clean = pd.DataFrame({
            "source_line": range(1, 101),
            "message": [f"unique message {i}" for i in range(100)],
            "target": [0] * 80 + [1] * 20,
        })

    def test_split_is_disjoint_stratified_and_reproducible(self):
        train, test = split_messages(self.clean)
        train_again, test_again = split_messages(self.clean)
        self.assertEqual((len(train), len(test)), (80, 20))
        self.assertEqual(int(train.target.sum()), 16)
        self.assertEqual(int(test.target.sum()), 4)
        self.assertFalse(set(train.source_line) & set(test.source_line))
        self.assertEqual(set(train.source_line) | set(test.source_line), set(self.clean.source_line))
        pd.testing.assert_frame_equal(train, train_again)
        pd.testing.assert_frame_equal(test, test_again)

    def test_duplicate_messages_are_rejected(self):
        self.clean.loc[1, "message"] = self.clean.loc[0, "message"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            split_messages(self.clean)
