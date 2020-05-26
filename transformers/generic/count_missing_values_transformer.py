"""Count of missing values per row"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class CountMissingPerRowTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        if X.ncols == 0:
            return np.zeros((X.nrows, 1))
        return X[:, dt.sum([dt.isna(dt.f[x]) for x in range(X.ncols)])]


class CountMissingNumericsPerRowTransformer(CountMissingPerRowTransformer):
    def transform(self, X: dt.Frame):
        return super().transform(X[:, [int, float]])


class CountMissingStringsPerRowTransformer(CountMissingPerRowTransformer):
    def transform(self, X: dt.Frame):
        return super().transform(X[:, [str]])
