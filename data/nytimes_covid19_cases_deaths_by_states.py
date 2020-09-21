"""Upload daily COVID-19 cases and deaths in US by states from NY Times github"""

# Contributors: Gregory Kanevsky - gregory@h2o.ai
# Created: July 9th, 2020
# Last Updated:


from typing import Union, List, Dict
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir
from datatable import f, g, join, by, sort, update, shift, isna


class NYTimesCovid19DailyCasesDeathsByStatesData(CustomData):
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
        group_by_cols = ["state"]
        forecast_len = 7

        # get COVID19 data from NYTimes github
        us_states = dt.fread("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv")

        # get states population
        us_states_pop = dt.fread(
            "http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")
        us_states_pop.names = {'NAME': 'state', 'POPESTIMATE2019': 'pop'}
        us_states_pop.key = "state"

        # augment data with state population figures and create adjusted case and death counts
        series_cols = ["cases", "deaths"]
        aggs = {f"{col}100k": dt.f[col] / (dt.g.pop / 100000) for col in series_cols}
        us_states[:, dt.update(pop=g.pop, pop100k=g.pop / 10000, **aggs), join(us_states_pop)]

        # remove rows without state defined (resulted in unmatched rows after left outer join)
        del us_states[isna(f.pop), :]

        # produce lag of 1 unit and add as new feature for each column in the list
        series_cols.extend([col + "100k" for col in series_cols])
        aggs = {f"{col}_yesterday": shift(f[col]) for col in series_cols}
        us_states[:, update(**aggs), sort(date_col), by(group_by_cols)]

        # update NA lags to 0
        aggs = {f"{col}_yesterday": 0 for col in series_cols}
        us_states[isna(f[f"{series_cols[0]}_yesterday"]), update(**aggs)]

        # compute daily values by differentiating
        aggs = {f"{col}_daily": f[col] - f[f"{col}_yesterday"] for col in series_cols}
        us_states[:, update(**aggs), sort(date_col), by(group_by_cols)]

        # delete columns with yesterday (shift) values
        series_cols_to_delete = [f"{col}_yesterday" for col in series_cols]
        del us_states[:, series_cols_to_delete]

        # set negative daily values to 0
        us_states[f.cases_daily < 0, [f.cases_daily, f.cases100k_daily]] = 0
        us_states[f.deaths_daily < 0, [f.deaths_daily, f.deaths100k_daily]] = 0

        # determine threshold to split train and test based on forecast horizon
        dates = dt.unique(us_states[:, date_col])
        split_date = dates[-(forecast_len + 1):, :, dt.sort(date_col)][0, 0]
        test_date = dates[-1, :, dt.sort(date_col)][0, 0]

        # split data to honor forecast horizon in test set
        df = us_states[date_col].to_pandas()
        train = us_states[df[date_col] <= split_date, :]
        test = us_states[df[date_col] > split_date, :]

        # return [train, test] and rename dataset names as needed
        return {f"covid19_daily_{split_date}_by_states_train": train,
                f"covid19_daily_{test_date}_by_states_test": test}
