"""First N chars of a string variable"""
import importlib
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np

class firstNChars(CustomTransformer):
    _numeric_output = False

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def firstN(s):
        # return first 4 characters. Change n to suit the problem
        n = 4
        nn = n+1
        return s[0:nn]

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).fillna("NA").iloc[:, 0].apply(lambda x: self.firstN(x))
