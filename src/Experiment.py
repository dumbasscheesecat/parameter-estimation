import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid 
from SignalDetection import SignalDetection 


class Experiment:
    def __init__(self):
        self.conditions = []

    def add_condition(self, sdt_obj: SignalDetection, label: str = None):  #Changed to make sure object is always instance of Signal Detection
        if not isinstance(sdt_obj, SignalDetection):
            raise TypeError("sdt_obj must be an instance of SignalDetection")
        self.conditions.append((sdt_obj, label)) 

    def sorted_roc_points(self):
        if not self.conditions:
            raise ValueError("No conditions added to the experiment")

        roc_points = [(sdt.false_alarm_rate(), sdt.hit_rate()) for sdt, _ in self.conditions]
        roc_points.sort(key=lambda x: x[0])

        false_alarm_rates, hit_rates = zip(*roc_points)
        return list(false_alarm_rates), list(hit_rates)

    def compute_auc(self):
        if not self.conditions:
            raise ValueError("No conditions added to compute AUC")

        false_alarm_rates, hit_rates = self.sorted_roc_points()
        return trapezoid(hit_rates, false_alarm_rates)  # Changed so scipy uses trapezoid instead of trapz which was not working 

    def plot_roc_curve(self, show_plot=True):
        false_alarm_rates, hit_rates = self.sorted_roc_points()

        plt.figure(figsize=(6, 6))
        plt.plot(false_alarm_rates, hit_rates, marker="o", linestyle="-", label="ROC Curve")
        plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC)") #Changed label
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        if show_plot:
            plt.show()

