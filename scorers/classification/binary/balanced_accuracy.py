"""Brier Loss"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder


class MyBrierLoss(CustomScorer):
    _description = "Balanced Accuracy Score"
    _binary = True
    _multiclass = True
    _maximize = True
    _perfect_score = 1
    _supports_sample_weight = True  
    _display_name = "Balanced Accuracy"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        return balanced_accuracy_score(actual, predicted, sample_weigh)
