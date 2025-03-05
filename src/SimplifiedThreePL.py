import numpy as np
import src.Experiment as experiment

class SimplifiedThreePL:
    def __init__(self, experiment):
        if isinstance(experiment, Experiment):
            raise TypeError("experiment must be an instance of Experiment")
        if not experiment.conditions:
            raise ValueError("experiment must contain at least one sdt condition")
        self.experiment=experiment

    def summary(self):
        n_correct=sum(sdt.hits for sdt, _ in self.experiment.conditions)
        n_incorrect=sum(sdt.misses + sdt.false_alarms for sdt, _ in self.experiment.conditions)
        return {
            "n_total": n_correct+n_incorrect,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": len(self.experiment.conditions)
        }
    
    def predict(self):