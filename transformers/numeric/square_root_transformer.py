"""Converts numbers to the square root, preserving the sign of the original numbers"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class SquareRootTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=3, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, [(dt.f[i]/dt.abs(dt.f[i]))*dt.exp(0.5*dt.log(dt.abs(dt.f[i]))) for i in range(X.ncols)]]
