"""Mean Average Precision @ k (MAP@k)"""

"""
Add value of k in recipe_dict in config or scorer will use default value of 10
If the number of classes is less than the chosen value, use number of classes instead.

Sample Datasets
# Expedia - https://www.kaggle.com/c/expedia-hotel-recommendations/overview/evaluation
recipe_dict = "{'k_for_map': 5}"
"""

import typing
import numpy as np

from sklearn.preprocessing import LabelEncoder
from h2oaicore.metrics import CustomScorer
from h2oaicore.systemutils import config


class MAPatk(CustomScorer):
    _description = "Mean Average Precision @ k (MAP@k)"
    _multiclass = True
    _maximize = True
    _perfect_score = 1
    _display_name = "MAP@k"
    _supports_sample_weight = False

    def score(self, actual: np.array, predicted: np.array, sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        num_classes = len(labels)
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        default_k = 10 if num_classes > 10 else num_classes
        k = config.recipe_dict['k_for_map'] if 'k_for_map' in config.recipe_dict else default_k
        best_k = np.argsort(predicted, axis=1)[:, -k:][:, ::-1]
        mapk = 0.
        for i in range(k):
            mapk += ((best_k[:, i] - actual) == 0).mean() / (i + 1)
        return mapk
