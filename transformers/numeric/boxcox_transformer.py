"""Box-Cox Transform"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from scipy.stats import boxcox


class BoxCoxTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        XX = X.to_pandas().iloc[:, 0].values
        is_na = np.isnan(XX)
        if not any(~is_na):
            return X
        self._offset = -XX.min() if XX.min() < 0 else 0
        self._offset += 1e-3
        self._lmbda = boxcox(self._offset + XX[~is_na], lmbda=None)[1]  # compute lambda
        return self.transform(X)

    def transform(self, X: dt.Frame):
        XX = X.to_pandas().iloc[:, 0].values
        is_na = np.isnan(XX) | np.array(XX <= -self._offset)
        if not any(~is_na):
            return X
        ret = boxcox(self._offset + XX[~is_na], lmbda=self._lmbda)  # apply transform with pre-computed lambda
        XX[~is_na] = ret
        return XX
