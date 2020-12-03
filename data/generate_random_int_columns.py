"""Data recipe to add one or more columns containing random integers."""

import datatable as dt
import numpy as np
import pandas as pd
from typing import Union, List, Dict
from h2oaicore.data import BaseData
from h2oaicore.data import CustomData


class GenerateRandomColumnsData(CustomData):

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
        col_count = 2
        col_names = ["random_col_1", "random_col_2"]

        if col_count != len(col_names):
            raise ValueError("Number of column names must be equal to number of columns.")

        if X is None:
            return []

        rcol = dt.Frame(np.random.randint(0, 100, size=(X.shape[0], col_count)))
        rcol.names = col_names
        X.cbind(rcol)

        return X
