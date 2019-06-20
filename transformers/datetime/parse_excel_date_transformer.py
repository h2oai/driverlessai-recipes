# Michelle Tanco - 05/21/2019
# Takes integers and decimals, treats them as an excel date sting, returns the date parts

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
import datetime as dttm


class ParseExcelDateTransformer(CustomTransformer):

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return False

    # Works on any number of numeric columns
    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    # Function for our training data sets
    def fit_transform(self, X: dt.Frame, y: np.array = None):

        df = X.to_pandas()

        # leave if any number is out of range of excel dates
        if df.iloc[:, 0].max() > 2958464:
            return np.zeros((X.shape[0], 5))

        df["new_date"] = pd.TimedeltaIndex(df.iloc[:, 0], unit='d') + dttm.datetime(1899, 12, 30)
        df["day"] = df["new_date"].dt.day
        df["month"] = df["new_date"].dt.month
        df["year"] = df["new_date"].dt.year
        df["hour"] = df["new_date"].dt.hour
        df["minute"] = df["new_date"].dt.minute

        # Re-Write to show WHAT each component is... or call DAI date transformers
        return df.iloc[:, 2:]

    # Function for validation and testing data sets
    def transform(self, X: dt.Frame):

        df = X.to_pandas()

        # leave if any number is out of range of excel dates
        if df.iloc[:, 0].max() > 2958464:
            return np.zeros((X.shape[0], 5))

        df["new_date"] = pd.TimedeltaIndex(df.iloc[:, 0], unit='d') + dttm.datetime(1899, 12, 30)
        df["day"] = df["new_date"].dt.day
        df["month"] = df["new_date"].dt.month
        df["year"] = df["new_date"].dt.year
        df["hour"] = df["new_date"].dt.hour
        df["minute"] = df["new_date"].dt.minute

        # Re-Write to show WHAT each component is... or call DAI date transformers
        return df.iloc[:, 2:]
