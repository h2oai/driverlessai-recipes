"""Adds together 3 or more numeric features"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class SumTransformer(CustomTransformer):
    
    _regression = True
    _binary = True
    _multiclass = True
    _numeric_output = True
    _is_reproducible = True
    _included_model_classes = None  # List[str]
    _excluded_model_classes = None  # List[str]

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=3, max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, dt.sum([dt.f[x] for x in range(X.ncols)])]
