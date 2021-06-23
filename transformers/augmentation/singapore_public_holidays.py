"""Flag for whether a date falls on a public holiday in Singapore."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd


# https://github.com/rjchow/singapore_public_holidays
def make_holiday_frame():
    return dt.fread(
        """
        Date,Name,Day,Observance,Observance Strategy
        2016-01-01,New Year's Day,Friday,2016-01-01,actual_day
        2016-02-08,Chinese New Year Day 1,Monday,2016-02-08,actual_day
        2016-02-09,Chinese New Year Day 2,Tuesday,2016-02-09,actual_day
        2016-03-25,Good Friday,Friday,2016-03-25,actual_day
        2016-05-01,Labour Day,Sunday,2016-05-01,next_monday
        2016-05-21,Vesak Day,Saturday,2016-05-21,actual_day
        2016-07-06,Hari Raya Puasa,Wednesday,2016-07-06,actual_day
        2016-08-09,National Day,Tuesday,2016-08-09,actual_day
        2016-09-12,Hari Raya Haji,Monday,2016-09-12,actual_day
        2016-10-29,Deepavali,Saturday,2016-10-29,actual_day
        2016-12-25,Christmas Day,Sunday,2016-12-26,next_monday
        2017-01-01,New Year's Day,Sunday,2017-01-02,next_monday
        2017-01-28,Chinese New Year Day 1,Saturday,2017-01-28,actual_day
        2017-01-29,Chinese New Year Day 2,Sunday,2017-01-30,next_monday
        2017-04-14,Good Friday,Friday,2017-04-14,actual_day
        2017-05-01,Labour Day,Monday,2017-05-01,actual_day
        2017-05-10,Vesak Day,Wednesday,2017-05-10,actual_day
        2017-06-25,Hari Raya Puasa,Sunday,2017-06-26,next_monday
        2017-08-09,National Day,Wednesday,2017-08-09,actual_day
        2017-09-01,Hari Raya Haji,Friday,2017-09-01,actual_day
        2017-10-18,Deepavali,Wednesday,2017-10-18,actual_day
        2017-12-25,Christmas Day,Monday,2017-12-25,actual_day
        2018-01-01,New Year's Day,Monday,2018-01-01,actual_day
        2018-02-16,Chinese New Year Day 1,Friday,2018-02-16,actual_day
        2018-02-17,Chinese New Year Day 2,Saturday,2018-02-17,actual_day
        2018-03-30,Good Friday,Friday,2018-03-30,actual_day
        2018-05-01,Labour Day,Tuesday,2018-05-01,actual_day
        2018-05-29,Vesak Day,Tuesday,2018-05-29,actual_day
        2018-06-15,Hari Raya Puasa,Friday,2018-06-15,actual_day
        2018-08-09,National Day,Thursday,2018-08-09,actual_day
        2018-08-22,Hari Raya Haji,Wednesday,2018-08-22,actual_day
        2018-11-06,Deepavali,Tuesday,2018-11-06,actual_day
        2018-12-25,Christmas Day,Tuesday,2018-12-25,actual_day
        2019-01-01,New Year's Day,Monday,2019-01-01,actual_day
        2019-02-05,Chinese New Year Day 1,Tuesday,2019-02-05,actual_day
        2019-02-06,Chinese New Year Day 2,Wednesday,2019-02-06,actual_day
        2019-04-19,Good Friday,Friday,2019-04-19,actual_day
        2019-05-01,Labour Day,Wednesday,2019-05-01,actual_day
        2019-05-19,Vesak Day,Sunday,2019-05-20,next_monday
        2019-06-05,Hari Raya Puasa,Wednesday,2019-06-05,actual_day
        2019-08-09,National Day,Friday,2019-08-09,actual_day
        2019-08-11,Hari Raya Haji,Sunday,2019-08-12,next_monday
        2019-10-27,Deepavali,Sunday,2019-10-27,next_monday
        2019-12-25,Christmas Day,Wednesday,2019-12-25,actual_day
        2020-01-01,New Year's Day,Wednesday,2020-01-01,actual_day
        2020-01-25,Chinese New Year Day 1,Saturday,2020-01-25,actual_day
        2020-01-26,Chinese New Year Day 2,Sunday,2020-01-27,next_monday
        2020-04-10,Good Friday,Friday,2020-04-10,actual_day
        2020-05-01,Labour Day,Friday,2020-05-01,actual_day
        2020-05-07,Vesak Day,Thursday,2020-05-07,actual_day
        2020-05-24,Hari Raya Puasa,Sunday,2020-05-25,next_monday
        2020-07-31,Hari Raya Haji,Friday,2020-07-31,actual_day
        2020-08-09,National Day,Sunday,2020-08-10,next_monday
        2020-11-14,Deepavali,Saturday,2020-11-14,actual_day
        2020-12-25,Christmas Day,Friday,2020-12-25,actual_day
        """).to_pandas()


class SingaporePublicHolidayTransformer(CustomTransformer):

    @staticmethod
    def get_default_properties():
        return dict(col_type="date", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.time_column = self.input_feature_names[0]
        hdays = make_holiday_frame()['Observance']
        self.memo = pd.DataFrame(hdays, columns=[self.time_column], dtype='datetime64[ns]')
        self.memo['year'] = self.memo[self.time_column].dt.year
        self.memo['doy'] = self.memo[self.time_column].dt.dayofyear
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
