"""Returns 1 if a value is missing, or 0 otherwise"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class FlagMissingTransformer(CustomTransformer):
    _unsupervised = True

    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, dt.isna(dt.f[0])]
