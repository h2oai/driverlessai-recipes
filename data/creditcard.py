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
        if 'default payment next month' in X.names:
            # e.g. train data
            X[:, 'default payment next month leak'] = X[:, 'default payment next month']
        else:
            # e.g. test data without target, for testing purposes, ensure still CC dataset
            cc_names = ["AGE", "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
                        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5",
                        "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
            assert all([x in cc_names for x in X.names])

        return X
