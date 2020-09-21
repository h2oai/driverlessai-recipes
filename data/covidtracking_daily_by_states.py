"""Upload daily Covid Tracking (https://covidtracking.com) US States
   cases, hospitalization, recovery, test and death data """

# Contributors: Gregory Kanevsky - gregory@h2o.ai
# Created: May 2d, 2020
# Last Updated: May 2d, 2020
# Reference: The Covid Tracking Project Data API
# API to retrieve state daily data: https://covidtracking.com/api

from typing import Union, List, Dict
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
import requests
from datatable import f, g, join, by, sort, update, shift, isna, count


class CovidtrackingDailyStateData(CustomData):
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

        # state codes lookup table
        us_state_codes = dt.Frame(code=['AL', 'AK', 'AS', 'AZ', 'AR', 'CA',
                                        'CO', 'CT', 'DE', 'DC', 'FL', 'GA',
                                        'GU', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY',
                                        'LA', 'ME', 'MD', 'MA', 'MI', 'MN',
                                        'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
                                        'NJ', 'NM', 'NY', 'NC', 'ND',
                                        'MP', 'OH', 'OK', 'OR', 'PA',
                                        'PR', 'RI', 'SC', 'SD', 'TN',
                                        'TX', 'UT', 'VT', 'VI', 'VA', 'WA',
                                        'WV', 'WI', 'WY'],
                                  state=['Alabama', 'Alaska', 'American Samoa', 'Arizona', 'Arkansas', 'California',
                                         'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida',
                                         'Georgia',
                                         'Guam', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
                                         'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
                                         'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
                                         'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
                                         'Northern Mariana Islands', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
                                         'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
                                         'Texas', 'Utah', 'Vermont', 'Virgin Islands', 'Virginia', 'Washington',
                                         'West Virginia', 'Wisconsin', 'Wyoming'])
        us_state_codes.key = "state"

        # get states population lookup table
        us_states_pop = dt.fread(
            "http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")
        us_states_pop.names = {'NAME': 'state', 'POPESTIMATE2019': 'pop'}
        us_states_pop = us_states_pop[dt.f.STATE > 0, :]
        us_states_pop.key = "state"

        # join state codes and population into single lookup table
        us_states_pop[:, dt.update(code=dt.g.code), dt.join(us_state_codes)]
        us_states_pop.key = "code"

        # US Covid Tracking API: https://covidtracking.com/data/api
        us_states = dt.fread("https://covidtracking.com/api/v1/states/daily.csv")
        # remove deprecated fields
        deprecated = ['checkTimeEt', 'commercialScore', 'dateChecked', 'dateModified', 'grade', 'hash',
                      'hospitalized', 'negativeIncrease', 'negativeRegularScore', 'negativeScore', 'posNeg',
                      'positiveScore', 'score', 'total']
        us_states = us_states[:, list(set(us_states.names) - set(deprecated))]
        us_states.names = {'state': 'code'}

        series_cols = ["positive", "negative", "hospitalizedCumulative", "inIcuCumulative",
                       "onVentilatorCumulative", "recovered", "death"]
        aggs = {f"{col}100k": f[col] / (g.pop / 100000) for col in series_cols}
        us_states[:, dt.update(state=g.state, pop=g.pop, pop100k=g.pop / 10000, **aggs), join(us_states_pop)]
        us_states = us_states[~dt.isna(dt.f.state), :]

        # produce lag of 1 unit and add as new feature for each shift column
        series_cols.extend([col + "100k" for col in series_cols])
        aggs = {f"{col}_yesterday": shift(f[col]) for col in series_cols}
        us_states[:, update(**aggs), sort(date_col), by(group_by_cols)]

        # update NA lags
        aggs = {f"{col}_yesterday": 0 for col in series_cols}
        us_states[isna(f[f"{series_cols[0]}_yesterday"]), update(**aggs)]

        aggs = {f"{col}_daily": f[col] - f[f"{col}_yesterday"] for col in series_cols}
        us_states[:, update(**aggs), sort(date_col), by(group_by_cols)]

        for col in series_cols:
            del us_states[:, f[f"{col}_yesterday"]]

        # validate dataset
        if us_states[:, count(), by(dt.f.state, f.date)][f.count > 1, :].shape[0] > 1:
            raise ValueError("Found duplicate elements for the same date and state.")

        # determine threshold to split train and test based on forecast horizon
        dates = dt.unique(us_states[:, date_col])
        split_date = dates[-(forecast_len + 1):, :, dt.sort(date_col)][0, 0]
        test_date = dates[-1, :, dt.sort(date_col)][0, 0]

        # split data to honor forecast horizon in test set
        df = us_states[date_col].to_pandas()
        train = us_states[df[date_col] <= split_date, :]
        test = us_states[df[date_col] > split_date, :]

        return {f"covidtracking_daily_{split_date}_by_us_states_train": train,
                f"covidtracking_daily_{test_date}_by_us_states_test": test}
