"""
Mean Average Precision @ 5 (MAP@5)
https://www.kaggle.com/c/expedia-hotel-recommendations/overview/evaluation
"""
import typing
import numpy as np
import pandas as pd
from h2oaicore.metrics import CustomScorer


class MAPat5(CustomScorer):
    _description = "Mean Average Precision @ 5 (MAP@5)"
    _multiclass = True
    _maximize = True
    _perfect_score = 1
    _display_name = "MAP@5"

    def score(self, actual: np.array, predicted: np.array, sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        predicted5 = [preds.argsort()[-5:][::-1] for preds in predicted]
        predicted_labels = [[labels[x] for x in preds] for preds in predicted5]
        df = pd.DataFrame.from_records(predicted_labels)
        map5 = map5eval(df, actual, len(labels))
        return map5


def map5eval(predicted, actual, n_classes):
    metric = 0.
    if n_classes > 4:
        for i in range(5):
            pred = pd.Series.tolist(predicted.iloc[:, i])
            metric += np.sum(actual == pred) / (i + 1)
        metric /= actual.shape[0]
    else:
        return 0.
    return metric
