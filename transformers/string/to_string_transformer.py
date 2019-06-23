"""Converts numbers to strings"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyToStringTransformer(CustomTransformer):
    _numeric_output = False

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
