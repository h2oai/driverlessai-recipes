"""Create airlines dataset"""
import uuid
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir


class AirlinesData(CustomData):
    # base_url = "http://stat-computing.org/dataexpo/2009/"  # used to work, but 404 now
    url = "https://0xdata-public.s3.amazonaws.com/data_recipes_data/1987.csv.bz2"

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config
        import bz2

        def extract_bz2(file, output_file):
            zipfile = bz2.BZ2File(file)
            data_file = zipfile.read()
            open(output_file, 'wb').write(data_file)

        temp_path = os.path.join(user_dir(), config.contrib_relative_directory, "airlines_%s" % str(uuid.uuid4()))
        os.makedirs(temp_path, exist_ok=True)

        link = AirlinesData.url
        file = download(link, dest_path=temp_path)
        output_file = file.replace(".bz2", "")
        print("%s %s" % (file, output_file))
        extract_bz2(file, output_file)

        return output_file
