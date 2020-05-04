"""Upload Monthly Seattle Rain Inches data set from data provided by the City of Seattle"""

# Contributors: Michelle Tanco - michelle.tanco@h2oai
# Created: October 18th, 2019
# Last Updated: October 18th, 2019


from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir


class SeattleRainDataRaw(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config

        # Location in DAI file system where we will save the data set
        temp_path = os.path.join(user_dir(), config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)

        # URL of desired data, this comes from the City of Seattle
        link = "https://data.seattle.gov/resource/rdtp-hzy3.csv"

        # Download the file
        file = download(link, dest_path=temp_path)

        # Give the file a descriptive name for the UI
        output_file = file.replace("rdtp-hzy3", "seattle_monthly_rain_raw")
        os.rename(file, output_file)

        # Return the location on the DAI server for this data set
        return output_file