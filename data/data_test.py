"""Create test dataset"""
import uuid
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir


class TestData(CustomData):
    url = "http://data.un.org/_Docs/SYB/CSV/SYB63_226_202009_Net%20Disbursements%20from%20Official%20ODA%20to%20Recipients.csv"

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

        link = TestData.url
        file = download(link, dest_path=temp_path)

        return file
