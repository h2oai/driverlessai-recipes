"""Custom RMSE Scorer that also gets X (original features) - for demo/testing purposes only"""
from h2oaicore.metrics import RmseScorer, CustomScorer
import numpy as np
import datatable as dt
import typing


class RmseWithXScorer(CustomScorer):
    _regression = True
    _needs_X = True
    _maximize = False

    def score(
            self,
            actual: np.array,
            predicted: np.array,
            sample_weight: typing.Optional[np.array] = None,
            labels: typing.Optional[np.array] = None,
            X: typing.Optional[dt.Frame] = None,
            **kwargs) -> float:
        assert X  # for testing
        return RmseScorer().score(actual, predicted, sample_weight, labels, **kwargs)
