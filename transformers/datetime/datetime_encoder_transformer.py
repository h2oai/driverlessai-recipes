"""Converts datetime column into an integer (milliseconds since 1970)"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd


class MyDateTimeTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="datetime", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = pd.to_datetime(X.to_pandas().iloc[:, 0], format=self.datetime_formats[self.input_feature_names[0]])
        return X.fillna(pd.Timedelta(seconds=0)).astype(np.int)
