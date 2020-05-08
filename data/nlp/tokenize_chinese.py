"""Chinese text tokenization using jieba package - https://github.com/fxsjy/jieba"""
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

cols_to_tokenize = []

_global_modules_needed_by_name = ["jieba==0.42.1"]

class TokenizeChiense(CustomData):

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        # exit gracefully if method is called as a data upload rather than data modify
        if X is None:
            return []
        # Tokenize the chinese text
        import jieba
        X = dt.Frame(X).to_pandas()
        # If no columns to tokenize, use the first column
        if len(cols_to_tokenize) == 0:
            cols_to_tokenize.append(X.columns[0])
        for col in cols_to_tokenize:
            X[col] = X[col].astype('unicode').fillna(u'NA')
            X[col] = X[col].apply(lambda x: " ".join([r[0] for r in jieba.tokenize(x)]))
        return dt.Frame(X)
