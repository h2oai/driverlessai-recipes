"""Cohen’s Kappa with linear weights"""
import typing

import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder


class QuadraticWeightedKappaScorer(CustomScorer):
    _description = "Cohen’s kappa with linear weights: a statistic that measures inter-annotator agreement."
    _multiclass = True
    _maximize = True
    _perfect_score = 1
    _display_name = "COHEN_KAPPA"

    def score(
        self,
        actual: np.array,
        predicted: np.array,
        sample_weight: typing.Optional[np.array] = None,
        labels: typing.Optional[np.array] = None,
        **kwargs
    ) -> float:

        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        predicted = np.argmax(predicted, axis=1)

        return cohen_kappa_score(
            actual,
            predicted,
            labels=labels,
            weights="linear",
            sample_weight=sample_weight,
        )
