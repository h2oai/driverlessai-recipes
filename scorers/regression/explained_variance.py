"""Explained Variance. Fraction of variance that is explained by the model."""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
import sklearn


class MyExplainedVarianceScorer(CustomScorer):
    _description = "My Explained Variance Scorer for Regression."
    _regression = True
    _maximize = True
    _perfect_score = 1
    _display_name = "ExpVar"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        return sklearn.metrics.explained_variance_score(actual, predicted, sample_weight=sample_weight,
                                                        multioutput='uniform_average')
