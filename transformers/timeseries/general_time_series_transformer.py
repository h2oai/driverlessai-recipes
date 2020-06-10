"""Demonstrates the API for custom time-series transformers."""
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np


class GeneralTimeSeriesTransformer(CustomTimeSeriesTransformer):
    _causal_recipe_allowed = False  # need self.encoder only available in lag time series recipe mode
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        # FIXME - use the following attributes
        # self.encoder
        # self.tgc
        # self.pred_gap
        # self.pred_periods
        # self.lag_sizes
        # self.lag_feature
        # self.target
        # self.tsp
        # self.time_column
        # self.datetime_formats
        self.encoder.fit(X[:, self.time_column].to_pandas())
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return self.encoder.transform(X[:, self.time_column].to_pandas())
