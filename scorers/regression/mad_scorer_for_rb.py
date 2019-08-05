"""MAD for RB"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import mean_absolute_error


class MyMadScoreForRB(CustomScorer):
    _description = "RB metric."
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "RB_MAD"
    _supports_sample_weight = False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:

        avg_pred = np.mean(predicted)

        if np.isclose(avg_pred, 0):
            return 100
        else:
            return 100 * mean_absolute_error(actual, predicted) / avg_pred
