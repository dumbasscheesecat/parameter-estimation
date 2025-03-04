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