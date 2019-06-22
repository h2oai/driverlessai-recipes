"""Returns a flag for whether a date falls on a holiday for each of Germany's Bundeslaender"""

from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np
import pandas as pd
import holidays


# Inherit from CustomTimeSeriesTransformer
# That means we only get the time_column, not any date/datetime column
class GermanyLandersHolidayTransformer(CustomTimeSeriesTransformer):
    _modules_needed_by_name = ['holidays']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = X[:, self.time_column]
        X = X.to_pandas()
        ge_holidays = holidays.DE()
        X["is_ge_holiday"] = X[self.time_column].apply(lambda x: x in ge_holidays)
        for prov in ["BW", 'BY', 'BE', 'BB', 'HB', 'HH', 'HE', 'MV', 'NI', 'NW', 'RP', 'SL', 'SN', 'ST', 'SH', 'TH']:
            ge_prov_holidays = holidays.DE(state=prov)
            X["is_ge_holiday_%s" % prov] = X[self.time_column].apply(lambda x: x in ge_prov_holidays)
        X.drop(self.time_column, axis=1, inplace=True)
        return X
