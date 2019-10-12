"""Create airlines dataset"""

import datatable as dt
from h2oaicore.stats import CustomData
import numpy as np

class AirlinesData(CustomData):
    @staticmethod
    def is_enabled():
        return False

    _display_name = "Airlines"
    _description = "Create airlines data from raw data"


def _create_data(input_file=""):
    import os
    from h2oaicore.systemutils_more import extract, download
    from h2oaicore.systemutils import config
    import shutil
    import bz2

    def extract_bz2(file, output_file):
        zipfile = bz2.BZ2File(file) # open the file
        data = zipfile.read() # get the decompressed data
        open(output_file, 'wb').write(data) # write a uncompressed file

    is_installed_path = os.path.join(config.data_directory, config.contrib_env_relative_directory, "airlines")
    is_installed_file = os.path.join(is_installed_path, "airlines_is_installed")

    temp_path = os.path.join(config.data_directory, config.contrib_relative_directory, "airlines")
    os.makedirs(temp_path, exist_ok=True)

    link = "http://stat-computing.org/dataexpo/2009/1987.csv.bz2"
    file = download(link, dest_path=temp_path)
    output_file = file.replace(".bz2", "")

    if not os.path.isfile(is_installed_file):
        extract_bz2(file, output_file)

        os.makedirs(is_installed_path, exist_ok=True)
        with open(is_installed_file, "wt") as f:
            f.write("DONE")

    return output_file
