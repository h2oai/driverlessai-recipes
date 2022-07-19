"""
Data Recipe to load XML datasets from a zip file. 
Just include this script inside the zip and upload it as a data recipe. 
"""

from typing import Union, List
from h2oaicore.data import CustomData
from h2oaicore.systemutils import user_dir, config
import datatable as dt
import numpy as np
import pandas as pd
import os
import glob
import uuid
from zipfile import ZipFile
import xml.etree.ElementTree as ET

FILE_EXTENSION = ".xml"

"""
This recipe expects the XML file to have a schema similar to the one below:
<data>
  <row>
    <col_1>Value</col_1>
    <col_2>Value</col_2>
  </row>
  <row>
    <col_1>Value</col_1>
    <col_2>Value</col_2>
  </row>
</data>

The <data> and <row> tags do not need to follow the same name convention. 
The col_1, col_2, ..., col_N tags will provide the column names for the final dataset.
"""


class XMLLoadFromZip(CustomData):
    @staticmethod
    def create_data(
        X: dt.Frame = None,
    ) -> Union[
        str,
        List[str],
        dt.Frame,
        List[dt.Frame],
        np.ndarray,
        List[np.ndarray],
        pd.DataFrame,
        List[pd.DataFrame],
    ]:
        zip_location = os.path.join(config.data_directory, "uploads")
        zip_files = glob.glob(os.path.join(zip_location, "*.zip"))
        if not zip_files:
            raise ValueError(
                f"No zip files found, please create a zip archive including"
                + f"all {FILE_EXTENSION} data files you want to load and this .py script."
            )
        latest_zip = max(zip_files, key=os.path.getctime)
        zip_fl = ZipFile(latest_zip)

        data_files = [
            f
            for f in zip_fl.namelist()
            if (f.endswith(FILE_EXTENSION) and not f.startswith("__MACOSX"))
        ]

        if data_files is None:
            return ValueError(f"No file with {FILE_EXTENSION} extension found!")

        temp_path = os.path.join(
            user_dir(),
            config.contrib_relative_directory,
            "extract_data_%s" % str(uuid.uuid4()),
        )

        os.makedirs(temp_path, exist_ok=True)

        for fl in data_files:
            zip_fl.extract(fl, path=temp_path)

        data_sets = {}

        for f in data_files:
            full_data_path = os.path.join(temp_path, f)

            if not os.path.exists(full_data_path):
                raise ValueError("File <<" + full_data_path + ">> does not exists!")

            with open(full_data_path, "r") as fl:
                root = ET.XML(fl.read())
                data = {}
                for row in root:
                    for item in row:
                        colname = item.tag
                        value = item.text
                        column = data.get(colname, [])
                        column.append(value)
                        data.update({colname: column})
                df = pd.DataFrame(data)
            data_sets.update({f: df})

        return data_sets
