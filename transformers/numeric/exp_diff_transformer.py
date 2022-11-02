"""Exponentiated difference of two numbers"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyExpDiffTransformer(CustomTransformer):
    _unsupervised = True

    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _interpretability = 10
    _interpretability_min = 3

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X_diff = X[:, dt.f[0] - dt.f[1]]
        X_diff[dt.f[0] > 20, 0] = 20  # want no more than e**20
        return X_diff[:, dt.exp(dt.f[0])]
