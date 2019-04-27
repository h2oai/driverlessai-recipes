from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd


class MyDateTimeTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="datetime", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._datetime_format = kwargs['datetime_format']

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return pd.to_datetime(X.to_pandas().iloc[:, 0], format=self._datetime_format).astype(np.int)
