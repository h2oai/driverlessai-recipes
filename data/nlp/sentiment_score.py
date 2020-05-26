"""Data recipe to get sentiment score using textblob"""
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

# text column name to get the sentiment score
text_colnames = ["text"]
# output dataset name
output_dataset_name = "df_with_sentiment"

_global_modules_needed_by_name = ["textblob"]


class SentimentScoreClass(CustomData):
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
        from textblob import TextBlob

        X = dt.Frame(X).to_pandas()
        for text_colname in text_colnames:
            X["sentiment_dai_" + text_colname] = X[text_colname].astype(str).fillna("NA").apply(
                lambda x: TextBlob(x).sentiment[0])

        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)

        # Save files to disk
        file_train = os.path.join(temp_path, output_dataset_name + ".csv")
        X.to_csv(file_train, index=False)

        return [file_train]
