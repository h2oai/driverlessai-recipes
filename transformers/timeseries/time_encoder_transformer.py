"""converts the Time Column to an ordered integer"""
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np


class MyTimeColTransformer(CustomTimeSeriesTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.encoder.fit(X.to_pandas())
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return self.encoder.transform(X.to_pandas())
