"""Returns the amount of US holidays for a given week"""
from h2oaicore.transformer_utils import CustomTransformer, convert_to_datetime
import datatable as dt
import numpy as np
import pandas as pd
import holidays


class HolidaysThisWeek(CustomTransformer):
    _unsupervised = True

    _modules_needed_by_name = ['holidays']
    _display_name = 'HolidaysThisWeek'

    @staticmethod
    def get_default_properties():
        return dict(col_type="date", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_column = self.input_feature_names[0]
        self.country = "US"

    def fit(self, X, y=None, **fit_params):

        self._output_feature_names = ["HolidaysThisWeek:%s" % self.time_column]
        self._feature_desc = ["Amount of holidays in current week for %s" % self.time_column]

        X = X[:, self.time_column].to_pandas()
        X[self.time_column] = convert_to_datetime(X[self.time_column], self.datetime_formats[self.time_column])

        mn_year = X[self.time_column].dt.year.min()
        mx_year = X[self.time_column].dt.year.max()
        if np.isnan(mn_year) or np.isnan(mx_year):
            years = []
        else:
            years = np.arange(int(mn_year), int(mx_year + mx_year - mn_year + 2))

        hdays = holidays.CountryHoliday(self.country)
        for year in list(years):
            hdays._populate(year)
        hdays.observed = True
        hdays = [date for date, name in sorted(hdays.items())]

        self.memo = pd.DataFrame(hdays, columns=[self.time_column], dtype='datetime64[ns]')
        self.memo['year'] = self.memo[self.time_column].dt.year
        self.memo['week'] = self.memo[self.time_column].dt.weekofyear
        self.memo.drop(self.time_column, axis=1, inplace=True)
        self.memo = self.memo.groupby(by=['year', 'week'], as_index=False).size()
        self.memo.rename(columns={'size': self._output_feature_names[0]}, inplace=True)
        return self

    def transform(self, X, **kwargs):
        X = X[:, self.time_column].to_pandas()
        X[self.time_column] = convert_to_datetime(X[self.time_column], self.datetime_formats[self.time_column])

        X['year'] = X[self.time_column].dt.year
        X['week'] = X[self.time_column].dt.weekofyear
        X.drop(self.time_column, axis=1, inplace=True)
        X = X.merge(self.memo, how='left', on=['year', 'week']).fillna(0)
        X = X[[self._output_feature_names[0]]].astype(int)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
