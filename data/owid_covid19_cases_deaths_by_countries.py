"""Upload daily COVID-19 cases and deaths by countries"""

# Contributors: Gregory Kanevsky - gregory@h2o.ai
# Created: August 29th, 2020
# Last Updated:


from typing import Union, List, Dict
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir
from datatable import f, g, join, by, sort, update, shift, isna


class OWIDCovid19DailyCasesDeathsByCountriesData(CustomData):
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

        # get COVID19 new cases data from Our World in Data github
        X = dt.fread("https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv")

        # remove country aggregates like 'World' and 'International'
        X = X[~(dt.f.iso_code == '') & ~(dt.f.continent == ''), :]

        # determine threshold to split train and test based on forecast horizon
        dates = dt.unique(X[:, date_col])
        split_date = dates[-(forecast_len + 1):, :, dt.sort(date_col)][0, 0]
        test_date = dates[-1, :, dt.sort(date_col)][0, 0]

        # split data to honor forecast horizon in test set
        train = X[dt.f[date_col] <= split_date, :]
        test = X[dt.f[date_col] > split_date, :]

        # return [train, test] and rename dataset names as needed
        return {f"covid19_daily_{split_date}_by_countries_train": train,
                f"covid19_daily_{test_date}_by_countries_test": test}
