"""Optimizes for specific Confusion Matrix Values: FP"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class CMFalsePositive(CustomScorer):
    _description = "Reduce false positive count"
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "FP"
    _threshold = 0.5

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        predicted = (predicted > self._threshold)
        cm = confusion_matrix(actual, predicted, sample_weight=sample_weight, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        return fp


