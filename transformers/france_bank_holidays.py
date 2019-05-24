from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np
import pandas as pd


# https://github.com/rjchow/singapore_public_holidays
def make_holiday_frame():
    return dt.fread(
"""
Date,Name,Day,Observance,Observance Strategy
2015-01-01,Nouvel an,Thursday,2015-01-01,actual_day
2015-04-06,Lundi de Paques,Monday,2015-04-06,actual_day
2015-05-01,Fete du travail,Friday,2015-05-01,actual_day
2015-05-08,8 mai 1945,Friday,2015-05-08,actual_day
2015-05-14,Jeudi de l'ascension,Thursday,2015-05-14,actual_day
2015-05-25,Lundi de Pentecote,Monday,2015-05-25,actual_day
2015-07-14,Fete nationale,Tuesday,2015-07-14,actual_day
2015-08-15,Assomption,Saturday,2015-08-15,actual_day
2015-11-01,La Toussaint,Sunday,2015-11-01,actual_day
2015-11-11,Armistice,Wednesday,2015-11-11,actual_day
2015-12-25,Noel,Friday,2015-12-25,actual_day
2016-01-01,Nouvel an,Friday,2016-01-01,actual_day
2016-03-28,Lundi de Paques,Monday,2016-03-28,actual_day
2016-05-01,Fete du travail,Sunday,2016-05-01,actual_day
2016-05-05,Jeudi de l'ascension,Thursday,2016-05-05,actual_day
2016-05-08,8 mai 1945,Sunday,2016-05-08,actual_day
2016-05-16,Lundi de Pentecote,Monday,2016-05-16,actual_day
2016-07-14,Fete nationale,Tuesday,2016-07-14,actual_day
2016-08-15,Assomption,Monday,2016-08-15,actual_day
2016-11-01,La Toussaint,Tuesday,2016-11-01,actual_day
2016-11-11,Armistice,Friday,2016-11-11,actual_day
2016-12-25,Noel,Sunday,2016-12-25,actual_day
2017-01-01,Nouvel an,Sunday,2017-01-01,actual_day
2017-04-17,Lundi de Paques,Monday,2017-04-17,actual_day
2017-05-01,Fete du travail,Monday,2017-05-01,actual_day
2017-05-08,8 mai 1945,Monday,2017-05-08,actual_day
2017-05-25,Jeudi de l'ascension,Thursday,2017-05-25,actual_day
2017-06-05,Lundi de Pentecote,Monday,2017-06-05,actual_day
2017-07-14,Fete nationale,Friday,2017-07-14,actual_day
2017-08-15,Assomption,Tuesday,2017-08-15,actual_day
2017-11-01,La Toussaint,Wednesday,2017-11-01,actual_day
2017-11-11,Armistice,Tuesday,2017-11-11,actual_day
2017-12-25,Noel,Monday,2017-12-25,actual_day
2018-01-01,Nouvel an,Friday,2018-01-01,actual_day
2018-04-17,Lundi de Paques,Monday,2018-04-17,actual_day
2018-05-01,Fete du travail,Sunday,2018-05-01,actual_day
2018-05-08,8 mai 1945,Sunday,2018-05-08,actual_day
2018-05-25,Jeudi de l'ascension,Thursday,2018-05-25,actual_day
2018-06-05,Lundi de Pentecote,Monday,2018-06-05,actual_day
2018-07-14,Fete nationale,Tuesday,2018-07-14,actual_day
2018-08-15,Assomption,Monday,2018-08-15,actual_day
2018-11-01,La Toussaint,Saturday,2018-11-01,actual_day
2018-11-11,Armistice,Sunday,2018-11-11,actual_day
2018-12-25,Noel,Sunday,2018-12-25,actual_day
2019-01-01,Nouvel an,Tuesday,2019-01-01,actual_day
2019-04-22,Lundi de Paques,Monday,2019-04-22,actual_day
2019-05-01,Fete du travail,Wednesday,2019-05-01,actual_day
2019-05-08,8 mai 1945,Wednesday,2019-05-08,actual_day
2019-05-30,Jeudi de l'ascension,Thursday,2019-05-30,actual_day
2019-06-10,Lundi de Pentecote,Monday,2019-06-10,actual_day
2019-07-14,Fete nationale,Sunday,2019-07-14,actual_day
2019-08-15,Assomption,Thursday,2019-08-15,actual_day
2019-11-01,La Toussaint,Friday,2019-11-01,actual_day
2019-11-11,Armistice,Monday,2019-11-11,actual_day
2019-12-25,Noel,Wednesday,2019-12-25,actual_day
2020-01-01,Nouvel an,Wednesday,2020-01-01,actual_day
2020-04-13,Lundi de Paques,Monday,2020-04-13,actual_day
2020-05-01,Fete du travail,Friday,2020-05-01,actual_day
2020-05-08,8 mai 1945,Friday,2020-05-08,actual_day
2020-05-21,Jeudi de l'ascension,Thursday,2020-05-21,actual_day
2020-06-01,Lundi de Pentecote,Monday,2020-06-01,actual_day
2020-07-14,Fete nationale,Tuesday,2020-07-14,actual_day
2020-08-15,Assomption,Saturday,2020-08-15,actual_day
2020-11-01,La Toussaint,Sunday,2020-11-01,actual_day
2020-11-11,Armistice,Wednesday,2020-11-11,actual_day
2020-12-25,Noel,Friday,2020-12-25,actual_day
""").to_pandas()


class FranceBankHolidayTransformer(CustomTimeSeriesTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        hdays = make_holiday_frame()['Observance']
        self.memo = pd.DataFrame(hdays, columns=[self.time_column], dtype='datetime64[ns]')
        self.memo['year'] = self.memo[self.time_column].dt.year
        self.memo['doy'] = self.memo[self.time_column].dt.dayofyear
        self.memo.sort_values(by=['year', 'doy']).drop_duplicates(subset=['year'], keep='first').reset_index(drop=True)
        self.memo.drop(self.time_column, axis=1, inplace=True)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = X[:, self.time_column]
        if X[:, self.time_column].ltypes[0] != dt.ltype.str:
            assert self.datetime_formats[self.time_column] in ["%Y%m%d", "%Y%m%d%H%M"]
            X[:, self.time_column] = dt.stype.str32(dt.stype.int64(dt.f[0]))
        X.replace(['', 'None'], None)
        X = X.to_pandas()
        X.loc[:, self.time_column] = pd.to_datetime(X[self.time_column],
                                                    format=self.datetime_formats[self.time_column])

        X['year'] = X[self.time_column].dt.year
        X['doy'] = X[self.time_column].dt.dayofyear
        X.drop(self.time_column, axis=1, inplace=True)
        feat = 'is_holiday'
        self.memo[feat] = 1
        X = X.merge(self.memo, how='left', on=['year', 'doy']).fillna(0)
        self.memo.drop(feat, axis=1, inplace=True)
        X = X[[feat]].astype(int)
        return X
