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


class ModelTests(unittest.TestCase):
    def test_vectorizer_does_not_learn_test_vocabulary(self):
        from src.model import train_model
        train = pd.DataFrame({"message": ["lunch meeting", "free prize", "hello friend", "claim reward"], "target": [0, 1, 0, 1]})
        model = train_model(train)
        before = dict(model.named_steps["vectorizer"].vocabulary_)
        model.predict(["testonlyword free"])
        self.assertNotIn("testonlyword", model.named_steps["vectorizer"].vocabulary_)
        self.assertEqual(before, model.named_steps["vectorizer"].vocabulary_)

    def test_error_orientation_and_metrics(self):
        import numpy as np
        from src.model import evaluate_model
        class FixedModel:
            classes_ = np.array([0, 1])
            def predict(self, messages):
                return np.array([0, 1, 0, 1, 1])
            def predict_proba(self, messages):
                return np.array([[.8, .2], [.2, .8], [.7, .3], [.1, .9], [.2, .8]])
        test = pd.DataFrame({"source_line": range(5), "message": list("abcde"), "target": [0, 0, 1, 1, 1], "label": ["ham", "ham", "spam", "spam", "spam"]})
        metrics, results = evaluate_model(FixedModel(), test)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 2]])
        self.assertEqual(metrics["accuracy"], 3/5)
        self.assertAlmostEqual(metrics["spam_precision"], 2/3)
        self.assertAlmostEqual(metrics["spam_recall"], 2/3)
        self.assertEqual(results.error_type.tolist(), ["correct", "false_positive", "false_negative", "correct", "correct"])
