"""Box-Cox Transform"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from scipy.stats import boxcox


class BoxCoxTransformer(CustomTransformer):
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
        x = self._offset + XX[~is_na]
        x = np.asarray(x)
        x[x <= 0] = 1e-3
        try:
            self._lmbda = boxcox(x, lmbda=self._lmbda)[1]  # compute lambda
        except ValueError as e:
            if 'Data must not be constant' in str(e):
                self._lmbda = None
                return X
            raise
        return self.transform(X)

    def transform(self, X: dt.Frame):
        XX = X.to_pandas().iloc[:, 0].values
        is_na = np.isnan(XX) | np.array(XX <= -self._offset)
        if not any(~is_na) or self._lmbda is None:
            return X
        x = self._offset + XX[~is_na]
        x = np.asarray(x)
        x[x <= 0] = 1e-3  # don't worry if not invertible, just ensure can transform and valid transforms are kept valid
        try:
            ret = boxcox(x, lmbda=self._lmbda)  # apply transform with pre-computed lambda
        except ValueError as e:
            if 'Data must not be constant' in str(e):
                return X
            raise
        XX[~is_na] = ret
        return XX
