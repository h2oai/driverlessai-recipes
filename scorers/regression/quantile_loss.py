"""Quantile Loss regression"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer


class QuantileLoss(CustomScorer):
    _alpha = 0.8  # CUSTOMIZE
    _description = "Quantile Loss for alpha=%g" % _alpha
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "Quantile alpha=%g" % _alpha
    _supports_sample_weight = True

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        q = QuantileLoss._alpha
        if sample_weight is None:
            sample_weight = np.ones(len(actual))
        return np.sum(
            sample_weight * np.maximum(q * (predicted - actual), (q - 1) * (predicted - actual))
        ) / np.sum(sample_weight)
