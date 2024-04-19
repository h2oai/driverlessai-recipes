"""Brier Loss"""
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
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        return brier_score_loss(actual, predicted, sample_weight=sample_weight, pos_label=labels[1])
