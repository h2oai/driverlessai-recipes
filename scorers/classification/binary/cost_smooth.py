"""Using hard-coded dollar amounts x for false positives and y for false negatives, calculate the cost of a model using: `(1 - y_true) * y_pred * fp_cost + y_true * (1 - y_pred) * fn_cost`"""

import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class CostBinary_smooth(CustomScorer):
    _description = "Calculates cost per row in binary classification: `(1 - y_true) * y_pred * fp_cost + y_true * (1 - y_pred) * fn_cost`"
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "Cost_smooth"

    # The cost of false positives and negatives will vary by data set, we use the rules from the below as an example
    # https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set
    _fp_cost = 75
    _fn_cost = 70

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        lb = LabelEncoder()
        labels = list(lb.fit_transform(labels))
        actual = lb.transform(actual)

        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])

        return np.sum(((1 - actual) * predicted * self.__class__._fp_cost + actual * (
                    1 - predicted) * self.__class__._fn_cost) * sample_weight) / np.sum(sample_weight)
