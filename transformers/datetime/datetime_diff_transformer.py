"""Difference in time between two datetime columns"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd


class MyDateTimeDiffTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="datetime", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        col1 = X.names[0]
        col2 = X.names[1]
        pd_t1 = X[:, col1].to_pandas().iloc[:, 0]
        pd_t2 = X[:, col2].to_pandas().iloc[:, 0]
        time1 = pd.to_datetime(pd_t1, format=self.datetime_formats[col1]).astype(np.int64)
        time2 = pd.to_datetime(pd_t2, format=self.datetime_formats[col2]).astype(np.int64)
        return time1 - time2
