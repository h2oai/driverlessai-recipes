"""Count of negative values per row"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class CountNegativePerRowTransformer(CustomTransformer):
    _unsupervised = True

    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols="all", max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, dt.rowsum(dt.f[:] < 0)]
