"""Optimizes for specific Confusion Matrix Values: `FN` - only recommended if threshold is adjusted"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class CMFalseNegative(CustomScorer):
    _threshold = 0.5   # Example only, should be adjusted based on domain knowledge and other experiments
    _description = "Reduce false negative count"
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "FN"

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
        return fn


