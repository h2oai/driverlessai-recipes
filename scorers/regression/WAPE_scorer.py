"""Weighted Absoluted Percent Error"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import median_absolute_error


class MyWAPE(CustomScorer):
    _description = "Weighted Absolute Percent Error. "
    _regression = True
    _maximize = False
    _perfect_score = 0
    _supports_sample_weight = False
    _display_name = "WAPE"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        return (abs(actual - predicted).sum() / actual.sum())
