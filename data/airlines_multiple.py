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
    base_url = "http://www.rdatasciencecases.org/Data/Airline/"

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
            data = zipfile.read()
            open(output_file, 'wb').write(data)

        temp_path = os.path.join(user_dir(), config.contrib_relative_directory, "airlines_%s" % str(uuid.uuid4()))
        os.makedirs(temp_path, exist_ok=True)

        link = AirlinesData.base_url + "1990.csv.bz2"
        file = download(link, dest_path=temp_path)
        output_file1 = file.replace(".bz2", "")
        print("%s %s" % (file, output_file1))
        extract_bz2(file, output_file1)

        link = AirlinesData.base_url + "1991.csv.bz2"
        file = download(link, dest_path=temp_path)
        output_file2 = file.replace(".bz2", "")
        print("%s %s" % (file, output_file2))
        extract_bz2(file, output_file2)

        return [output_file1, output_file2]
