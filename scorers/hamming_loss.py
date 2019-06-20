"""Hamming Loss - Misclassification Rate (1 - Accuracy)"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import LabelEncoder


class MyHammingLoss(CustomScorer):
    _description = "Hamming Loss (Misclassification Rate)"
    _multiclass = True
    _maximize = False
    _perfect_score = 0
    _display_name = "HAMMING"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        predicted = np.argmax(predicted, axis=1)
        return hamming_loss(actual, predicted, labels, sample_weight)

