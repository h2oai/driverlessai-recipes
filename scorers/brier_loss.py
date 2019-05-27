import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder


class MyBrierLoss(CustomScorer):
    _description = "Brier Score Loss"
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "BRIER"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        if labels is not None:
            actual = LabelEncoder().fit(labels).transform(actual)
        else:
            actual = LabelEncoder().fit_transform(actual)
        return brier_score_loss(actual, predicted, sample_weight)