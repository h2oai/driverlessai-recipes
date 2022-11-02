"""Converts numbers to the square root, preserving the sign of the original numbers"""
import math

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class SquareRootTransformer(CustomTransformer):
    _unsupervised = True

    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=3, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = X[:, [(dt.f[i] / dt.abs(dt.f[i])) * dt.exp(0.5 * dt.log(dt.abs(dt.f[i]))) for i in range(X.ncols)]]
        # Don't leave inf/-inf
        for i in range(X.ncols):
            X.replace([math.inf, -math.inf], None)
        return X
