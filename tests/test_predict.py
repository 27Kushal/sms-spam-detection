"""Prediction checks for inputs, probability labels and unknown vocabulary."""
import unittest
import numpy as np
import pandas as pd
from src.model import train_model
from src.predict import predict_sms, format_prediction


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_model(pd.DataFrame({"message": ["hello friend", "meet lunch", "free prize", "win reward"], "target": [0, 0, 1, 1]}))

    def test_blank_input_is_rejected(self):
        for message in ["", " \t", None]:
            with self.assertRaisesRegex(ValueError, "nonblank"):
                predict_sms(self.model, message)

    def test_known_message_probabilities_sum_to_one(self):
        result = predict_sms(self.model, "free prize")
        self.assertEqual(result["prediction"], "Spam")
        self.assertAlmostEqual(result["p_spam"] + result["p_ham"], 1)
        self.assertTrue(result["has_known_words"])

    def test_unknown_tokens_report_prior_only_prediction(self):
        result = predict_sms(self.model, "zzzxxyyqq")
        self.assertFalse(result["has_known_words"])
        self.assertIn("priors only", format_prediction(result))
        self.assertAlmostEqual(result["p_spam"], 0.5)

    def test_class_order_is_read_from_model(self):
        class ReverseOrder:
            classes_ = np.array([1, 0])
            named_steps = self.model.named_steps
            def predict_proba(self, messages):
                return np.array([[.8, .2]])
            def predict(self, messages):
                return np.array([1])
        result = predict_sms(ReverseOrder(), "free prize")
        self.assertEqual(result["p_spam"], .8)
        self.assertEqual(result["p_ham"], .2)
