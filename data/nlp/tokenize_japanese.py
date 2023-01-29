"""
Japanese text tokenization using janome package - https://mocobeta.github.io/janome/en/
"""
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

cols_to_tokenize = []  # Column name list to tokenize

_global_modules_needed_by_name = ["janome"]


class TokenizeJapanese(CustomData):

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        # exit gracefully if method is called as a data upload rather than data modify
        if X is None:
            return []
        # Tokenize the Japanese text
        from janome.tokenizer import Tokenizer
        t = Tokenizer(wakati=True)
        X = dt.Frame(X).to_pandas()
        # If no columns to tokenize, use the first column
        if len(cols_to_tokenize) == 0:
            cols_to_tokenize.append(X.columns[0])
        for col in cols_to_tokenize:
            X[col] = X[col].fillna("")  # replacing nan to str
            X[col] = X[col].apply(lambda x: " ".join([tkn for tkn in t.tokenize(x)]))
        return dt.Frame(X)
