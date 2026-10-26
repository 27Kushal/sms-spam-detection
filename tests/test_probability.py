"""Independent small examples for the assignment's probability definitions."""
import unittest
import numpy as np
import pandas as pd
from src.probability import analyze_probabilities


class ProbabilityTests(unittest.TestCase):
    def test_whole_word_presence_and_bayes(self):
        data = pd.DataFrame({"message": ["FREE free prize!", "Free time", "freedom winning", "win now"], "target": [1, 0, 0, 1]})
        words, summary = analyze_probabilities(data)
        free = words.set_index("word").loc["free"]
        self.assertEqual(free["messages_with_word"], 2)
        self.assertEqual(free["spam_with_word"], 1)
        self.assertEqual(free["p_spam_given_word"], 0.5)
        self.assertEqual(free["bayes_p_spam_given_word"], 0.5)
        self.assertEqual(words.set_index("word").loc["win", "messages_with_word"], 1)
        self.assertEqual(summary["expectation_Y"], 0.5)
        self.assertEqual(summary["variance_Y"], 0.25)
        self.assertEqual(summary["empirical_population_variance_Y"], 0.25)

    def test_absent_word_is_undefined(self):
        words, _ = analyze_probabilities(pd.DataFrame({"message": ["hello", "prize"], "target": [0, 1]}))
        free = words.set_index("word").loc["free"]
        self.assertTrue(np.isnan(free["p_spam_given_word"]))
        self.assertTrue(np.isnan(free["bayes_p_spam_given_word"]))

    def test_no_spam_class_is_handled(self):
        words, summary = analyze_probabilities(pd.DataFrame({"message": ["free time"], "target": [0]}))
        self.assertEqual(summary["variance_Y"], 0)
        self.assertEqual(words.set_index("word").loc["free", "p_spam_given_word"], 0)
        self.assertTrue(np.isnan(words.set_index("word").loc["free", "p_word_given_spam"]))

    def test_empty_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            analyze_probabilities(pd.DataFrame(columns=["message", "target"]))


if __name__ == "__main__":
    unittest.main()
