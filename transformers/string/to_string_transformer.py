"""Converts numbers to strings"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyToStringTransformer(CustomTransformer):
    _unsupervised = True

    _numeric_output = False
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @property
    def display_name(self):
        return "Str"

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).apply(lambda x: "str_" + x)
