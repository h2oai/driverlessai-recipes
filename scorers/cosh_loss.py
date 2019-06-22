"""Hyperbolic Cosine Loss"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
import sklearn


class CoshLossScorer(CustomScorer):
    _description = "Cosh loss for Regression"
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "COSH"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])
        good_rows = predicted >= 0
        if good_rows == 0:
            return 30
        delta = predicted[good_rows] - actual[good_rows]
        sample_weight = sample_weight[good_rows]
        loss = np.log1p(np.cosh(delta))
        return np.sum(sample_weight * loss) / np.sum(sample_weight)

