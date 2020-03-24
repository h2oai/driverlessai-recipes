"""Modify credit card dataset"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd


class CreditCardData(CustomData):

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:

        if X is None:
            return []
        X[:, 'default payment next month leak'] = X[:, 'default payment next month']

        return X
