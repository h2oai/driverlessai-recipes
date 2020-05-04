"""Transpose the Monthly Seattle Rain Inches data set for Time Series use cases"""

# Contributors: Michelle Tanco - michelle.tanco@h2oai
# Created: October 18th, 2019
# Last Updated: October 18th, 2019

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd


# This should be called from the Data Details page of the Seattle Monthly Rain data set within DAI
# This code could be included in the SeattleRainDataRaw method, but is shown as Modify code as an example
from virtualenv import user_dir


class SeattleRainDataClean(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        # exit gracefully if method is called as a data upload rather than data modify
        if X is None:
            return []
        import os
        from h2oaicore.systemutils import config

        # Change to pandas -> we can rewrite this as dt at a later date
        rain_raw = X.to_pandas()

        # Set index and pivot the data
        # Rows go from one row each month to one row each month & gauge
        rain_raw = rain_raw.set_index("date")
        rain_pivot = rain_raw.unstack().reset_index(name="rain_inches")
        rain_pivot.rename(columns={'level_0': 'rain_gauge', 'date': 'end_of_month'}, inplace=True)

        # Format date appropriately
        rain_pivot['end_of_month'] = pd.to_datetime(rain_pivot['end_of_month'])

        # Split data into train and test by date
        # Train on 7 years of data, test on 1 year of data
        train_py = rain_pivot[
            (rain_pivot['end_of_month'] >= '2009-01-01') & (rain_pivot['end_of_month'] <= '2016-01-01')]
        test_py = rain_pivot[rain_pivot['end_of_month'].dt.year == 2016]

        # Set up to save to disk
        temp_path = os.path.join(user_dir(), config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)

        # Save files to disk
        file_train = os.path.join(temp_path, "seattle_rain_train.csv")
        train_py.to_csv(file_train)
        file_test = os.path.join(temp_path, "seattle_rain_test.csv")
        test_py.to_csv(file_test)

        return [file_train, file_test]

        # Instead of saving to disk we could return our data frames instead
        # by writing the files we can return multiple and control their names in the UI
        # return rain_pivot # return a single data frame



