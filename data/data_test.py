"""Create test dataset"""
import uuid
import zipfile
from typing import Union, List

import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.data import CustomData
from h2oaicore.systemutils import user_dir


class TestData(CustomData):
    url = "http://archive.ics.uci.edu/static/public/53/iris.zip"
    dataset_name = "iris.data"

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

        zip_file = download(TestData.url, dest_path=temp_path)
        with zipfile.ZipFile(zip_file, "r") as my_zip:
            my_zip.extract(TestData.dataset_name, temp_path)

        return os.path.join(temp_path, TestData.dataset_name)
