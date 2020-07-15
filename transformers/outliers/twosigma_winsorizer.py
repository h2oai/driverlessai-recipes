"""Winsorizes (truncates) univariate outliers outside of two standard deviations from the mean."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyTwoSigmaWinsorizer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.mean = X.mean1()
        self.sd = X.sd1()
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = dt.Frame(X)
        if self.sd is not None and self.mean is not None:
            X[self.mean - 2 * self.sd > dt.f[0], float] = self.mean - 2 * self.sd
            X[self.mean + 2 * self.sd < dt.f[0], float] = self.mean + 2 * self.sd
        return X
