"""Returns a flag for whether a date falls on Ramadan in Saudi Arabia"""
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np
import pandas as pd


# from https://github.com/rjchow/singapore_public_holidays
def get_ramadan_dates():
    return dt.fread(
        """	
        DateStart,DateEnd
        2014-06-28,2014-07-28
        2015-06-17,2015-07-16
        2016-06-06,2016-07-05
        2017-05-26,2017-06-24
        2018-05-16,2018-06-14
        2019-05-05,2019-06-03
        2020-04-23,2020-05-23
        """).to_pandas()


class RamadanTransformer(CustomTimeSeriesTransformer):
    """
    Create feature 'is_ramadan' for ramadan days 
    Data are initialized for Saudi Arabia. Modify dates in get_ramadan_dates for other countries
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ramadan_dates = get_ramadan_dates()
        hramadan = pd.DataFrame([])
        for i in range(len(ramadan_dates)):
            hdates = pd.date_range(start=ramadan_dates['DateStart'][i], end=ramadan_dates['DateEnd'][i])
            if i == 0:
                hramadan = hdates
            else:
                hramadan = hramadan.append(hdates)
        self.memo = pd.DataFrame(hramadan, columns=[self.time_column], dtype='datetime64[ns]')
        self.memo['year'] = self.memo[self.time_column].dt.year
        self.memo['doy'] = self.memo[self.time_column].dt.dayofyear
        self.memo.drop(self.time_column, axis=1, inplace=True)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = X[:, self.time_column]
        if X[:, self.time_column].ltypes[0] != dt.ltype.str:
            assert self.datetime_formats[self.time_column] in ["%Y%m%d", "%Y%m%d%H%M", "%Y", "%Y%m"]
            X[:, self.time_column] = dt.stype.str32(dt.stype.int64(dt.f[0]))
        X.replace(['', 'None'], None)
        X = X.to_pandas()
        X.loc[:, self.time_column] = pd.to_datetime(X[self.time_column],
                                                    format=self.datetime_formats[self.time_column])
        X['year'] = X[self.time_column].dt.year
        X['doy'] = X[self.time_column].dt.dayofyear
        X.drop(self.time_column, axis=1, inplace=True)
        feat = 'is_ramadan'
        self.memo[feat] = 1
        X = X.merge(self.memo, how='left', on=['year', 'doy']).fillna(0)
        self.memo.drop(feat, axis=1, inplace=True)
        X = X[[feat]].astype(int)
        return X
