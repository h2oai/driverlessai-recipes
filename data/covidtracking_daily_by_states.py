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
        forecast_len = 7

        # state codes lookup table
        us_state_codes = dt.Frame(code = ['AL', 'AK', 'AS', 'AZ', 'AR', 'CA',
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
                                          'Colorado', 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
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
        us_states_pop[:, dt.update(code = dt.g.code), dt.join(us_state_codes)]
        us_states_pop.key = "code"

        # US Covid Tracking API: https://covidtracking.com/data/api
        us_states = dt.fread("https://covidtracking.com/api/v1/states/daily.csv")
        # remove deprecated fields
        deprecated = ['checkTimeEt', 'commercialScore', 'dateChecked', 'dateModified', 'grade', 'hash',
                          'hospitalized', 'negativeIncrease', 'negativeRegularScore', 'negativeScore', 'posNeg',
                          'positiveScore', 'score', 'total']
        us_states = us_states[:, list(set(us_states.names) - set(deprecated))]
        us_states.names = {'state': 'code'}

        us_states[:, dt.update( state=dt.g.state, pop=dt.g.pop, pop100k=dt.g.pop / 100000,
                                positive100k=dt.f.positive / (dt.g.pop / 100000),
                                negative100k=dt.f.negative / (dt.g.pop / 100000),
                                hospitalizedCumulative100k=dt.f.hospitalizedCumulative / (dt.g.pop / 100000),
                                inIcuCumulative100k=dt.f.inIcuCumulative / (dt.g.pop / 100000),
                                onVentilatorCumulative100k=dt.f.onVentilatorCumulative / (dt.g.pop / 100000),
                                recovered100k=dt.f.recovered / (dt.g.pop / 100000),
                                death100k=dt.f.death / (dt.g.pop / 100000)), dt.join(us_states_pop)]
        us_states = us_states[~dt.isna(dt.f.state), :]

        # validate dataset
        if us_states[:, dt.count(), dt.by(dt.f.state, dt.f.date)][dt.f.count>1, :].shape[0] > 1:
            raise ValueError("Found duplicate elements for the same date and state.")

        # determine threshold to split train and test based on forecast horizon
        dates = dt.unique(us_states[:, date_col])
        split_date = dates[-forecast_len:, :, dt.sort(date_col)][0, 0]

        # split data to honor forecast horizon in test set
        df = us_states[date_col].to_pandas()
        train = us_states[df[date_col] < split_date, :]
        test = us_states[df[date_col] >= split_date, :]

        return {"covidtracking_daily_us_states_train": train, "covidtracking_daily_us_states_test": test}