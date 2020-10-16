"""Upload daily COVID-19 cases and deaths in US total from NY Times github"""

# Contributors: Gregory Kanevsky - gregory@h2o.ai
# Created: October 15th, 2020
# Last Updated:


from typing import Union, List, Dict
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir
from datatable import f, g, join, by, sort, update, shift, isna


class NYTimesCovid19DailyCasesDeathsUSData(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[
        str, List[str],
        dt.Frame, List[dt.Frame],
        np.ndarray, List[np.ndarray],
        pd.DataFrame, List[pd.DataFrame],
        Dict[str, str],  # {data set names : paths}
        Dict[str, dt.Frame],  # {data set names : dt frames}
        Dict[str, np.ndarray],  # {data set names : np arrays}
        Dict[str, pd.DataFrame],  # {data set names : pd frames}
    ]:
        # define date column and forecast horizon
        date_col = 'date'
        forecast_len = 7

        # get COVID19 data from NYTimes github
        us_total = dt.fread("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv")

        # produce lag of 1 unit and add as new feature for each column in the list
        series_cols = ["cases", "deaths"]
        aggs = {f"{col}_yesterday": shift(f[col]) for col in series_cols}
        us_total[:, update(**aggs), sort(date_col)]

        # update NA lags to 0
        aggs = {f"{col}_yesterday": 0 for col in series_cols}
        us_total[isna(f[f"{series_cols[0]}_yesterday"]), update(**aggs)]

        # compute daily values by differentiating
        aggs = {f"{col}_daily": f[col] - f[f"{col}_yesterday"] for col in series_cols}
        us_total[:, update(**aggs), sort(date_col)]

        # delete columns with yesterday (shift) values
        series_cols_to_delete = [f"{col}_yesterday" for col in series_cols]
        del us_total[:, series_cols_to_delete]

        # set negative daily values to 0
        us_total[f.cases_daily < 0, [f.cases_daily]] = 0
        us_total[f.deaths_daily < 0, [f.deaths_daily]] = 0

        # determine threshold to split train and test based on forecast horizon
        dates = dt.unique(us_total[:, date_col])
        split_date = dates[-(forecast_len + 1):, :, dt.sort(date_col)][0, 0]
        test_date = dates[-1, :, dt.sort(date_col)][0, 0]

        # split data to honor forecast horizon in test set
        df = us_total[date_col].to_pandas()
        train = us_total[df[date_col] <= split_date, :]
        test = us_total[df[date_col] > split_date, :]

        # return [train, test] and rename dataset names as needed
        return {f"covid19_daily_{split_date}_us_train": train,
                f"covid19_daily_{test_date}_us_test": test}
