"""Data recipe to get summary of text using gensim"""
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

# text column name to get the summary
text_colnames = ["text"]
# output dataset name
output_dataset_name = "data_with_summary"

_global_modules_needed_by_name = ["gensim==4.3.2"]


class TextSummarizationClass(CustomData):
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
        from gensim.summarization.summarizer import summarize

        def summarize_paragraph(txt):
            try:
                return summarize(txt, ratio=0.5)
            except:
                return txt

        X = dt.Frame(X).to_pandas()
        for text_colname in text_colnames:
            X["summary_dai_" + text_colname] = X[text_colname].astype(str).fillna("NA").apply(
                lambda x: summarize_paragraph(x))

        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)

        # Save files to disk
        file_train = os.path.join(temp_path, output_dataset_name + ".csv")
        X.to_csv(file_train, index=False)

        return [file_train]
