from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np


class GeneralTimeSeriesTransformer(CustomTimeSeriesTransformer):
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        # FIXME - use the following attributes
        # self._encoder
        # self._tgc
        # self._pred_gap
        # self._pred_periods
        # self._lag_sizes
        # self._lag_feature
        # self._target
        # self._tsp
        # self._time_column
        # self._datetime_formats
        self._encoder.fit(X.to_pandas())
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return self._encoder.transform(X.to_pandas())

