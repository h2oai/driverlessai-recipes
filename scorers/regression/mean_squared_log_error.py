"""Mean Squared Log Error for regression"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import mean_squared_log_error


class MyMeanSquaredLogError(CustomScorer):
    _description = "My Mean Squared Error Scorer for Regression."
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "MSLE"
    _supports_sample_weight = False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        return mean_squared_log_error(actual, predicted)
