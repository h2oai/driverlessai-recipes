"""Median Absolute Error for predictions in the top decile"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer


class MyTopQuartileMedianAbsErrorScorer(CustomScorer):
    _description = "Median Abs Error for Top Decile"
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "TopDecile"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])
        cutoff = np.quantile(predicted, 0.9)
        which = (predicted >= cutoff).ravel()
        return float(np.median(np.abs(actual[which] - predicted[which]) * sample_weight[which]))
