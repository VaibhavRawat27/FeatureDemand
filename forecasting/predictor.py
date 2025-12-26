import numpy as np

class DemandPredictor:
    def predict(self, demands, similarities):
        weights = similarities / similarities.sum()
        return np.dot(weights, demands)
