"""Create multiple dataset test"""
import uuid
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir


class TestDataMultiple(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config

        temp_path = os.path.join(user_dir(), config.contrib_relative_directory, "testdata_%s" % str(uuid.uuid4()))
        os.makedirs(temp_path, exist_ok=True)

        link = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        output_file1 = download(link, dest_path=temp_path)

        link = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/bezdekIris.data"
        output_file2 = download(link, dest_path=temp_path)

        return [output_file1, output_file2]
