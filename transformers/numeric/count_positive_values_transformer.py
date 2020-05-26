"""Count of positive values per row"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class CountPositivePerRowTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols="all", max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, dt.sum([(dt.f[x] > 0) for x in range(X.ncols)])]
