"""Winsorizes (truncates) univariate outliers outside of a given quantile threshold"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyQuantileWinsorizer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"quantile": [0.01, 0.001, 0.05]}

    @property
    def display_name(self):
        return "MyQuantileWinsorizer%s" % str(self._quantile)

    def __init__(self, quantile=0.01, **kwargs):
        super().__init__(**kwargs)
        self._quantile = min(quantile, 1 - quantile)
        self._lo = None
        self._hi = None

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        vals = X.to_numpy()
        self._lo = float(np.quantile(vals, self._quantile))
        self._hi = float(np.quantile(vals, 1 - self._quantile))
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = dt.Frame(X)
        X[self._lo > dt.f[0], float] = self._lo
        X[self._hi < dt.f[0], float] = self._hi
        return X
