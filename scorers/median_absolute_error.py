"""Median Absolute Error for regression"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import median_absolute_error

class MyMedianAbsoluteError(CustomScorer):
    _description = "My Median Absolute Error Scorer for Regression."
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "MEDAE"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        return median_absolute_error(actual, predicted)

