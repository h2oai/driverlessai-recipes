"""Mean Absolute Scaled Error for time-series regression"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer


class MyMeanAbsoluteScaledErrorScorer(CustomScorer):
    _description = "My Mean Absolute Scaled Error for Time Series Regression."
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "MASE"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])

        naive_errors = np.abs(actual * sample_weight).mean()
        errors = np.abs((actual - predicted) * sample_weight)
        return errors.mean() / naive_errors
