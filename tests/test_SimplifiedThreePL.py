import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment
from src.SignalDetection import SignalDetection

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        correct_counts = [55, 60, 75, 90, 95]
        incorrect_counts = [45, 40, 25, 10, 5]
       
        conditions = []
        for correct, incorrect in zip(correct_counts, incorrect_counts):
            conditions.append((SignalDetection(correct, incorrect, 0, 0), f"Condition {len(conditions)}"))
       
        self.experiment = Experiment([])
        for sdt, label in conditions:
            self.experiment.add_condition(sdt, label)
       
        self.model = SimplifiedThreePL(self.experiment)

    def test_summary(self):
        summary = self.model.summary()
        self.assertEqual(summary["n_correct"], 375)
        self.assertEqual(summary["n_total"], 500)

    def test_fit(self):
        self.model.fit()
        self.assertTrue(self.model._is_fitted)

    def test_get_discrimination(self):
        self.model.fit()
        self.assertIsInstance(self.model.get_discrimination(), float)

    def test_prediction_values(self):
        self.model.fit()
        probs = self.model.predict([self.model.get_discrimination(), self.model.get_base_rate()])
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))

if __name__ == '__main__':
    unittest.main()
