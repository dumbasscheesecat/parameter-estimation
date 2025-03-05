import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        """Initialize test data and model instance."""
        self.difficulty_params = np.array([2, 1, 0, -1, -2])
        self.n_correct = np.array([55, 60, 75, 90, 95])
        self.n_incorrect = 100 - self.n_correct
        self.experiment = Experiment(self.difficulty_params, self.n_correct, self.n_incorrect)
        self.model = SimplifiedThreePL(self.experiment)
    
    def test_initialization(self):
        """Test proper initialization and attribute handling."""
        self.assertEqual(self.model._is_fitted, False)
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()
    
    def test_predict_output_range(self):
        """Test that predictions are in [0,1] range."""
        parameters = [1.0, 0.0]
        predictions = self.model.predict(parameters)
        self.assertTrue(np.all(predictions >= 0) and np.all(predictions <= 1))
    
    def test_base_rate_influence(self):
        """Test that higher base rate values lead to higher predicted probabilities."""
        a = 1.0
        q_low = -2.0  # Low base rate
        q_high = 2.0  # High base rate
        p_low = self.model.predict([a, q_low])
        p_high = self.model.predict([a, q_high])
        self.assertTrue(np.all(p_high > p_low))
    
    def test_difficulty_influence(self):
        """Test that higher difficulty values lower probability when a is positive."""
        a = 1.0
        q = 0.0
        p = self.model.predict([a, q])
        self.assertTrue(np.all(np.diff(p) > 0))  # Higher difficulty -> lower probability
    
    def test_ability_influence(self):
        """Test that higher ability leads to higher probabilities when a is positive."""
        a = 1.0
        q = 0.0
        self.model.experiment.difficulty_params = np.array([-2, -1, 0, 1, 2])  # Reverse difficulty
        p = self.model.predict([a, q])
        self.assertTrue(np.all(np.diff(p) > 0))  # Higher ability -> higher probability
    
    def test_likelihood_improvement(self):
        """Test that likelihood improves after fitting."""
        initial_guess = [1.0, 0.0]
        initial_nll = self.model.negative_log_likelihood(initial_guess)
        self.model.fit()
        final_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), np.log(self.model.get_base_rate() / (1 - self.model.get_base_rate()))])
        self.assertTrue(final_nll < initial_nll)
    
    def test_discrimination_estimate(self):
        """Test that a larger estimate of a is returned for steeper curves."""
        self.model.fit()
        est_a = self.model.get_discrimination()
        self.assertTrue(est_a > 0)  # Expect positive discrimination for increasing accuracy rates
    
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
        expected_accuracy = self.n_correct / (self.n_correct + self.n_incorrect)
        np.testing.assert_allclose(predictions, expected_accuracy, atol=0.1)
    
    def test_corruption_prevention(self):
        """Ensure users cannot create an inconsistent object."""
        with self.assertRaises(ValueError):
            corrupted_exp = Experiment([2, 1], [50], [50])  # Mismatched lengths
        with self.assertRaises(AttributeError):
            self.model._is_fitted = True  # Attempt to manually modify private attribute

if __name__ == '__main__':
    unittest.main()
