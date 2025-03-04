from scipy.stats import norm

class SignalDetection:
    def __init__(self, hits, misses, false_alarms, correct_rejections):
        self.hits = hits
        self.misses = misses
        self.false_alarms = false_alarms
        self.correct_rejections = correct_rejections

    def hit_rate(self):
        denominator = self.hits + self.misses
        return self.hits / denominator if denominator > 0 else 0  #Chatgpt gave fix to this code to avoid the error when division by zero occured

    def false_alarm_rate(self):
        denominator = self.false_alarms + self.correct_rejections
        return self.false_alarms / denominator if denominator > 0 else 0  #Read above

    def d_prime(self):
        return norm.ppf(self.hit_rate()) - norm.ppf(self.false_alarm_rate())

    def criterion(self):
        return -0.5 * (norm.ppf(self.hit_rate()) + norm.ppf(self.false_alarm_rate()))
