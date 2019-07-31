"""Log Mean Absolute Error for regression"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import mean_absolute_error


class LogMeanAbsoluteError(CustomScorer):
    _description = "Log Mean Absolute Error Scorer for Regression."
    _regression = True
    _maximize = False
    _perfect_score = -99.0
    _display_name = "LogMAE"
    _supports_sample_weight = False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        return np.log(mean_absolute_error(actual, predicted))
