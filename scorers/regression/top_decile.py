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
    _supports_sample_weight = False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        cutoff = np.quantile(predicted, 0.9)
        which = (predicted >= cutoff).ravel()
        if any(which):
            # must have one entry at least, else np.median([]) will give nan
            return float(np.median(np.abs(actual[which] - predicted[which])))
        else:
            # constant of some other case when no 90% quantile, just use all values
            return float(np.median(np.abs(actual - predicted)))
