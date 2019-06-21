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
        loss = np.log1p(np.cosh(predicted - actual))
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])
        return np.sum(sample_weight * loss) / np.sum(sample_weight)
