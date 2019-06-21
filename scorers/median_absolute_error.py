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
    _supports_sample_weight = False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not implemented for %s" % self.display_name)
        return median_absolute_error(actual, predicted)
