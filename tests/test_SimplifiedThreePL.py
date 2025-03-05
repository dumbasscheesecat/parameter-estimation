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

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        """Initialize test data and model instance."""
        self.experiment = Experiment()
        self.model = SimplifiedThreePL(self.experiment)
    
    def test_initialization(self):
        """Test proper initialization and attribute handling."""
        self.assertEqual(self.model._is_fitted, False)
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()
    
    def test_summary(self):
        """Test the summary method returns the correct structure."""
        summary = self.model.summary()
        self.assertIn('n_total', summary)
        self.assertIn('n_correct', summary)
        self.assertIn('n_incorrect', summary)
        self.assertIn('n_conditions', summary)
    
    def test_predict_output_range(self):
        """Test that predictions are in [0,1] range."""
        parameters = [1.0, 0.0]
        predictions = self.model.predict(parameters)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))
    
    def test_negative_log_likelihood(self):
        """Test that negative log-likelihood returns a valid numerical value."""
        parameters = [1.0, 0.0]
        nll = self.model.negative_log_likelihood(parameters)
        self.assertTrue(isinstance(nll, float))
    
    def test_fit(self):
        """Test that fit() runs without errors and sets parameters."""
        self.model.fit()
        self.assertTrue(self.model._is_fitted)
        self.assertTrue(isinstance(self.model.get_discrimination(), float))
        self.assertTrue(isinstance(self.model.get_base_rate(), float))
    
    def test_parameter_estimation(self):
        """Test that model estimates reasonable parameters after fitting."""
        self.model.fit()
        discrimination = self.model.get_discrimination()
        base_rate = self.model.get_base_rate()
        self.assertTrue(0 <= base_rate <= 1)  # Base rate should be in [0,1]
        self.assertTrue(discrimination > 0)  # Discrimination should be positive
    
    def test_stability_of_fit(self):
        """Test that fit() results in stable parameter estimates across multiple runs."""
        self.model.fit()
        a1, c1 = self.model.get_discrimination(), self.model.get_base_rate()
        self.model.fit()
        a2, c2 = self.model.get_discrimination(), self.model.get_base_rate()
        self.assertAlmostEqual(a1, a2, places=3)
        self.assertAlmostEqual(c1, c2, places=3)
    
    def test_integration(self):
        """Test fitting to predefined dataset and comparing predicted probabilities."""
        self.model.fit()
        predictions = self.model.predict([self.model.get_discrimination(), np.log(self.model.get_base_rate() / (1 - self.model.get_base_rate()))])
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))  # Ensure predictions are valid

if __name__ == '__main__':
    unittest.main()
