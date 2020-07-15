"""F3 Score"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import fbeta_score


class fbeta(CustomScorer):
    _description = "Fbeta(3))"
    _binary = True
    _multiclass = True
    _maximize = True
    _perfect_score = 1
    _display_name = "F3 Score"
    _threshold = 0.5  # Example only, should be adjusted based on domain knowledge and other experiments

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        method = "binary"
        if len(labels) > 2:
            predicted = np.argmax(predicted, axis=1)
            method = "micro"
        else:
            predicted = (predicted > self._threshold)
        f3_score = fbeta_score(actual, predicted, labels=labels, average=method, sample_weight=sample_weight, beta=3)
        return f3_score
