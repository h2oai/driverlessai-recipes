"""
Mean Average Precision @ k (MAP@k)
https://www.kaggle.com/c/expedia-hotel-recommendations/overview/evaluation
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

    def score(self, actual: np.array, predicted: np.array, sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        k = config.recipe_dict['k_for_map'] if 'k_for_map' in config.recipe_dict else 5
        predictedk = [preds.argsort()[-k:][::-1] for preds in predicted]
        predicted_labels = [[labels[x] for x in preds] for preds in predictedk]
        df = pd.DataFrame.from_records(predicted_labels)
        mapk = mapkeval(df, actual, len(labels), k)
        return mapk


def mapkeval(predicted, actual, n_classes, k):
    metric = 0.
    if n_classes > k-1:
        for i in range(k):
            pred = pd.Series.tolist(predicted.iloc[:, i])
            metric += np.sum(actual == pred) / (i + 1)
        metric /= actual.shape[0]
    else:
        return 0.
    return metric