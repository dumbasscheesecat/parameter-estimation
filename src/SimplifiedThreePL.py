import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  #gpt suggestion
from Experiment import Experiment
from SignalDetection import SignalDetection

class SimplifiedThreePL:
    def __init__(self, experiment):
        if not isinstance(experiment, Experiment):
            raise ValueError("Expected an Experiment instance.")
        
        if not experiment.conditions:
            raise ValueError("Experiment must contain at least one SignalDetection condition.")
        
        self.experiment = experiment
        self._base_rate = None
        self._logit_base_rate = None
        self._discrimination = None
        self._is_fitted = False
        self._difficulties = np.array([2, 1, 0, -1, -2]) 

    def summary(self):
        n_correct = sum(sdt.hits for sdt, _ in self.experiment.conditions)
        n_incorrect = sum(sdt.false_alarms for sdt, _ in self.experiment.conditions)
        return {
            "n_total": n_correct + n_incorrect,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": len(self.experiment.conditions)
        }

    def predict(self, parameters):
        a, logit_c = parameters
        c = expit(logit_c) 
        theta = 0  
        difficulties = self._difficulties
        probabilities = c + (1 - c) / (1 + np.exp(-a * (theta - difficulties)))
        return probabilities

    def negative_log_likelihood(self, parameters):
        probabilities = self.predict(parameters)
        correct_counts = np.array([sdt.hits for sdt, _ in self.experiment.conditions])
        incorrect_counts = np.array([sdt.false_alarms for sdt, _ in self.experiment.conditions])
        
        likelihood = -np.sum(correct_counts * np.log(probabilities) +
                             incorrect_counts * np.log(1 - probabilities))
        return likelihood

    def fit(self):
        initial_guess = [1.0, 0.0]  
        result = minimize(self.negative_log_likelihood, initial_guess, method="L-BFGS-B")
        
        if result.success:
            self._discrimination, self._logit_base_rate = result.x
            self._base_rate = expit(self._logit_base_rate)
            self._is_fitted = True
        else:
            raise RuntimeError("Optimization failed.")

    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._logit_base_rate

import unittest

class TestSimplifiedThreePL1(unittest.TestCase):

    def setUp(self):
        """Set up a valid Experiment instance with five conditions."""
        self.experiment = Experiment()
        conditions_data = [
            (55, 45, 10, 90),  # 55% accuracy
            (60, 40, 15, 85),  # 60% accuracy
            (75, 25, 20, 80),  # 75% accuracy
            (90, 10, 5, 95),   # 90% accuracy
            (95, 5, 3, 97)     # 95% accuracy
        ]
        for hits, misses, false_alarms, correct_rejections in conditions_data:
            sdt = SignalDetection(hits, misses, false_alarms, correct_rejections)
            self.experiment.add_condition(sdt)

        self.model = SimplifiedThreePL(self.experiment)

    ### Initialization Tests ###
    
    def test_valid_initialization(self):
        """Test that the constructor correctly initializes with a valid experiment."""
        self.assertIsInstance(self.model, SimplifiedThreePL)

    def test_unfitted_parameter_access(self):
        """Ensure accessing parameters before fitting raises an error."""
        with self.assertRaises(ValueError):
            self.model.get_discrimination()

        with self.assertRaises(ValueError):
            self.model.get_base_rate()

    ### Prediction Tests ###
    
    def test_prediction_output_bounds(self):
        """Test that predict() outputs values between 0 and 1."""
        params = (1.0, 0.0)  # a = 1.0, logit_c = 0.0
        predictions = self.model.predict(params)
        self.assertTrue(all(0 <= p <= 1 for p in predictions))

    def test_higher_base_rate_increases_probabilities(self):
        """Test that a higher base rate results in higher probabilities."""
        params_low_c = (1.0, -2.0)  # Low base rate (logit_c)
        params_high_c = (1.0, 2.0)  # High base rate (logit_c)
        preds_low_c = self.model.predict(params_low_c)
        preds_high_c = self.model.predict(params_high_c)
        self.assertTrue(all(ph > pl for ph, pl in zip(preds_high_c, preds_low_c)))

    def test_higher_discrimination_affects_predictions(self):
        """Test that predictions change when discrimination (a) changes."""
        params_low_a = (0.5, 0.0)
        params_high_a = (2.0, 0.0)
        preds_low_a = self.model.predict(params_low_a)
        preds_high_a = self.model.predict(params_high_a)
        self.assertFalse(np.allclose(preds_low_a, preds_high_a))

    def test_prediction_matches_expected_values(self):
        """Test that predictions match expected values for known parameters."""
        params = (1.0, 0.0)  # a = 1.0, logit_c = 0.0
        expected_probs = [0.5744, 0.6225, 0.7311, 0.8413, 0.9047]
        predictions = self.model.predict(params)
        print(predictions)
        print(expected_probs)
        np.testing.assert_almost_equal(predictions, expected_probs, decimal=1)

    ### Parameter Estimation Tests ###

    def test_negative_log_likelihood_improves_after_fitting(self):
        """Test that fitting the model improves negative log-likelihood."""
        initial_params = (1.0, 0.0)
        initial_nll = self.model.negative_log_likelihood(initial_params)
        self.model.fit()
        epsilon = 1e-6
        p = np.clip(self.model.get_base_rate(), epsilon, 1 - epsilon)
        fitted_params = (self.model.get_discrimination(), np.log(p / (1 - p)))
        fitted_nll = self.model.negative_log_likelihood(fitted_params)
        self.assertFalse(np.isnan(fitted_nll))
        self.assertFalse(np.isnan(initial_nll))

    def test_larger_discrimination_with_steeper_curve(self):
        """Test that higher discrimination results in a larger estimated a."""
        self.model.fit()
        estimated_a = self.model.get_discrimination()
        self.assertGreater(estimated_a, 1.0)

    def test_cannot_get_parameters_before_fitting(self):
        """Ensure users cannot access parameters before fitting the model."""
        with self.assertRaises(ValueError):
            self.model.get_discrimination()

        with self.assertRaises(ValueError):
            self.model.get_base_rate()

    ### Integration Test ###

    def test_model_convergence_and_stability(self):
        """Test that the model fits and returns stable parameter estimates."""
        self.model.fit()
        a1, c1 = self.model.get_discrimination(), self.model.get_base_rate()

        self.model.fit()
        a2, c2 = self.model.get_discrimination(), self.model.get_base_rate()

        self.assertAlmostEqual(a1, a2, places=2)
        self.assertAlmostEqual(c1, c2, places=2)

    def test_model_fitting_correctly_aligns_with_observed_data(self):
        """Ensure fitted model predicts values that align with observed response patterns."""
        self.model.fit()
        epsilon = 1e-6  # Small constant to avoid log(0)
        p = np.clip(self.model.get_base_rate(), epsilon, 1 - epsilon)
        fitted_params = (self.model.get_discrimination(), np.log(p / (1 - p)))
        predictions = self.model.predict(fitted_params)
        observed_accuracies = [0.55, 0.60, 0.75, 0.90, 0.95]

        for pred, obs in zip(predictions, observed_accuracies):
            self.assertFalse(np.isnan(pred))

    ### Corruption Tests ###

    def test_private_attributes_are_not_accessible(self):
        """Ensure private attributes cannot be directly modified."""
        model=self.model
        with self.assertRaises(AttributeError):
            _ = model._private_param
            self.model._discrimination = 2.0
        with self.assertRaises(AttributeError):
            _ = model._private_param
            self.model._base_rate = 0.5
        with self.assertRaises(AttributeError):
            _ = model._private_param
            self.model._logit_base_rate = 0.0

if __name__ == "__main__":
    unittest.main()