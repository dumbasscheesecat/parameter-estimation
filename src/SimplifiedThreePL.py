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

