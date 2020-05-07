"""Quantile Loss regression"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer


class QuantileLoss(CustomScorer):
    _quantile = 0.8  # CUSTOMIZE
    _description = "Quantile Loss for alpha=%g" % _quantile
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "Quantile alpha=%g" % _quantile
    _supports_sample_weight = True

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        q = QuantileLoss._quantile
        if sample_weight is None:
            sample_weight = np.ones(len(actual))
        return np.sum(
            sample_weight * np.maximum(q * (predicted - actual), (q - 1) * (predicted - actual))
        ) / np.sum(sample_weight)
