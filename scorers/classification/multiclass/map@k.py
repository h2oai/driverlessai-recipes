"""
Mean Average Precision @ k (MAP@k)
"""

"""
Add value of k in recipe_dict in config or scorer will use default value of 10
If the number of classes is less than the chosen value, use number of classes instead.

Sample Datasets
# Expedia - https://www.kaggle.com/c/expedia-hotel-recommendations/overview/evaluation
recipe_dict = "{'k_for_map': 5}"
"""

import typing
import numpy as np
import pandas as pd

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
              labels: typing.Optional[np.array] = None) -> float:
        num_classes = len(labels)
        default_k = 10 if num_classes > 10 else num_classes
        k = config.recipe_dict['k_for_map'] if 'k_for_map' in config.recipe_dict else default_k
        predictedk = [preds.argsort()[-k:][::-1] for preds in predicted]
        predicted_labels = [[labels[x] for x in preds] for preds in predictedk]
        df = pd.DataFrame.from_records(predicted_labels)
        mapk = mapkeval(df, actual, num_classes, k)
        return mapk


def mapkeval(predicted, actual, n_classes, k):
    metric = 0.
    for i in range(k):
        pred = pd.Series.tolist(predicted.iloc[:, i])
        metric += np.sum(actual == pred) / (i + 1)
    metric /= actual.shape[0]
    return metric