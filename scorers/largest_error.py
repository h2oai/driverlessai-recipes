"""Largest error for regression problems. Highly sensitive to outliers."""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer


class MyLargestErrorScorer(CustomScorer):
    _description = "My Largest Error Scorer for Regression."
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "LargestError"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])
        return (np.abs(actual - predicted) * sample_weight).max()
