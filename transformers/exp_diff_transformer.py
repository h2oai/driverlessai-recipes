from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyExpDiffTransformer(CustomTransformer):
    _interpretability = 10
    _interpretability_min = 3

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, dt.exp(dt.f[0] - dt.f[1])]

