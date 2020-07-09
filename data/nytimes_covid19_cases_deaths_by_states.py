"""Upload daily COVID-19 cases and deaths in US by states from NY Times github"""

# Contributors: Gregory Kanevsky - gregory@h2o.ai
# Created: July 9th, 2020
# Last Updated:


from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir


class NYTimesCovid19DailyCasesDeathsByStatesData(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        # define date column and forecast horizon
        date_col = 'date'
        forecast_len = 7

        # get COVID19 data from NYTimes github
        us_states = dt.fread("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv")

        # get states population
        us_states_pop = dt.fread(
            "http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv")
        us_states_pop.names = {'NAME': 'state', 'POPESTIMATE2019': 'pop'}
        us_states_pop.key = "state"

        # augment data with state population figures and create adjusted case and death counts
        us_states[:, dt.update(pop=dt.g.pop, pop100k=dt.g.pop / 100000,
                               cases100k=dt.f.cases / (dt.g.pop / 100000),
                               deaths100k=dt.f.deaths / (dt.g.pop / 100000)),
        dt.join(us_states_pop)]

        # determine threshold to split train and test based on forecast horizon
        dates = dt.unique(us_states[:, date_col])
        split_date = dates[-forecast_len:, :, dt.sort(date_col)][0, 0]

        # split data to honor forecast horizon in test set
        df = us_states[date_col].to_pandas()
        train = us_states[df[date_col] < split_date, :]
        test = us_states[df[date_col] >= split_date, :]

        # return [train, test] and rename dataset names as needed
        return {"covid19_daily_by_states_train": train, "covid10_daily_by_states_test": test}
