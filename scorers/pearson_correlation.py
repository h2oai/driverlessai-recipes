"""Pearson Correlation Coefficient for regression"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer


class Pearson_Correlation(CustomScorer):
    _description = "Pearson Correlation coefficient"
    _regression = True
    _maximize = True
    _perfect_score = 1.
    _display_name = "PearsonR"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])

        sx = np.sum(predicted * sample_weight)
        sy = np.sum(actual * sample_weight)
        sx_2 = np.sum(predicted * predicted * sample_weight)
        sy_2 = np.sum(actual * actual * sample_weight)
        sxy = np.sum(predicted * actual * sample_weight)
        n = np.sum(sample_weight)
        sq = np.sqrt(np.abs(n * sx_2 - (sx * sx)) * np.abs(n * sy_2 - (sy * sy)))
        cor = (n * sxy - sx * sy) / max(sq, 1E-30)

        return cor
