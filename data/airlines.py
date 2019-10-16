"""Create airlines dataset"""

import datatable as dt
from h2oaicore.data import CustomData
import numpy as np

class AirlinesData(CustomData):
    @staticmethod
    def create_data(data: dt.Frame = ""):
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config
        import bz2

        def extract_bz2(file, output_file):
            zipfile = bz2.BZ2File(file)
            data_file = zipfile.read()
            open(output_file, 'wb').write(data_file)

        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory, "airlines")
        os.makedirs(temp_path, exist_ok=True)

        link = "http://stat-computing.org/dataexpo/2009/1987.csv.bz2"
        file = download(link, dest_path=temp_path)
        output_file = file.replace(".bz2", "")
        print("%s %s" % (file, output_file))
        extract_bz2(file, output_file)

        return output_file
