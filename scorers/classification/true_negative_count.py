"""Optimizes for specific Confusion Matrix Values: TP, TN, FP, or FN"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

class CMTrueNegative(CustomScorer):

    _description = "Increase true negative count"
    _binary = True
    _multiclass = True
    _maximize = True
    _perfect_score = 10000000000
    _display_name = "True Negative Count"
    _threshold = 0.5

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:

        # label actual values
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)

        if len(labels) > 2:
            predicted = np.argmax(predicted, axis=1)
        else:
            predicted = (predicted > self._threshold)

        # use sklean to get values
        cm = confusion_matrix(actual, predicted, sample_weight=sample_weight, labels=labels)
        tn, fp, fn, tp = cm.ravel()

        return tn


