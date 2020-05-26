"""Check and match a list of words from a specific string column"""
"""The custom recipe example has used 'train_movie_sentiment.tsv' dataset"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd


class MatchWordsTransformer(CustomData):

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        if X is None:
            return []
        # Define here manually the list of the word to be matched
        _words_list = ['manager', 'Mamet', 'references', 'LOGO', 'Cynthia', 'Monstervision']

        data = X.to_pandas()

        data['keywords'] = ''
        words_list = [x.lower() for x in _words_list]
        for w in words_list:
            # Change the 'review' column to the column with the free text
            data['keywords'] = np.where(data['review'].str.lower().str.contains(w, regex=False), w, data['keywords'])
        data['keywords'] = np.where(data['keywords'] == '', np.nan, data['keywords'])

        return data
