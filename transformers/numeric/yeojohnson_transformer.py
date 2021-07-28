"""Yeo-Johnson Power Transformer"""
import math

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from scipy.stats import yeojohnson


class YeoJohnsonTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        XX = X.to_pandas().iloc[:, 0].values
        is_na = np.isnan(XX)
        self._offset = -np.nanmin(XX) if np.nanmin(XX) < 0 else 0
        self._offset += 1e-3
        self._lmbda = None
        if not any(~is_na):
            return X
        self._lmbda = yeojohnson(self._offset + XX[~is_na], lmbda=self._lmbda)[1]  # compute lambda
        return self.transform(X)

    def transform(self, X: dt.Frame):
        XX = X.to_pandas().iloc[:, 0].values
        is_na = np.isnan(XX) | np.array(XX <= -self._offset)
        if not any(~is_na) or self._lmbda is None:
            return X
        ret = yeojohnson(self._offset + XX[~is_na], lmbda=self._lmbda)  # apply transform with pre-computed lambda
        XX[~is_na] = ret
        XX = dt.Frame(XX)
        # Don't leave inf/-inf
        for i in range(X.ncols):
            XX.replace([math.inf, -math.inf], None)
        return XX
