import numpy as np
import scipy.optimize as optim
from scipy.special import expit #gpt suggestion
import src.Experiment as experiment

class SimplifiedThreePL:
    def __init__(self, experiment):
        if isinstance(experiment, Experiment):
            raise TypeError("experiment must be an instance of Experiment")
        if not experiment.conditions:
            raise ValueError("experiment must contain at least one sdt condition")
        self.experiment=experiment
        self._base_rate=None
        self._logit_base_rate=None
        self._discrimination=None
        self._is_fitted=False
        self._difficulties = np.array([2, 1, 0, -1, -2])

    def summary(self):
        n_correct=sum(sdt.hits for sdt, _ in self.experiment.conditions)
        n_incorrect=sum(sdt.misses + sdt.false_alarms for sdt, _ in self.experiment.conditions)
        return {
            "n_total": n_correct+n_incorrect,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": len(self.experiment.conditions)
        }
    
    def predict(self, parameters):
        a, q=parameters #gpt suggestion
        c=expit(q) 
        difficulties=self._difficulties
        exponent=a*(0-difficulties)
        probability_correct=c + (1 - c) * (1 / (1 + np.exp(-exponent)))
        return probability_correct

