"""Create Bosch competition datasets with leak"""

## Bosch Production Line Performance - Kaggle

##  1) Download train and test data from Slack public URLs
##  2) Unzip .zip files
##  3) Combine train and test data
##  4) Create leak features for train and test data based on row ids and row order
##  5) Import the data into Driverless AI for further experimentation

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd


class BoschData(CustomData):

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        # import packages
        import os
        import gc
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config
        import zipfile

        # define constants
        train_data_url = "https://files.slack.com/files-pri/T0329MHH6-F012UF3T2J0/download/bosch_train_full.zip?pub_secret=c59d0f381a"
        test_data_url = "https://files.slack.com/files-pri/T0329MHH6-F013ES4F6N4/download/bosch_test_full.zip?pub_secret=8726e8b7e2"

        # function for unzipping data
        def extract_zip(file, output_directory):
            with zipfile.ZipFile(file, "r") as zip_ref:
                zip_ref.extractall(output_directory)

        # download and unzip files
        temp_path = os.path.join(config.data_directory, "recipe_tmp", "bosch")
        os.makedirs(temp_path, exist_ok=True)

        for link in [train_data_url, test_data_url]:
            raw_file = download(link, dest_path=temp_path)
            extract_zip(raw_file, temp_path)

        # parse with datatable
        train_path = os.path.join(temp_path, "bosch_train_full.csv")
        test_path = os.path.join(temp_path, "bosch_test_full.csv")

        X_train = dt.fread(train_path)
        X_test = dt.fread(test_path)

        # add leak features
        train = X_train[:, ["Id", "Response"]].to_pandas()
        test = X_test[:, ["Id"]].to_pandas()

        date_features = [colname for colname in X_test.names if "D" in colname]

        train["Min_Date"] = X_train[:, date_features].to_pandas().min(axis=1).values
        test["Min_Date"] = X_test[:, date_features].to_pandas().min(axis=1).values

        ntrain = train.shape[0]
        train_test = pd.concat([train, test]).reset_index(drop=True)

        train_test.sort_values(by=["Min_Date", "Id"], ascending=True, inplace=True)

        train_test["Leak_1"] = train_test["Id"].diff()
        train_test["Leak_2"] = train_test["Id"].iloc[::-1].diff()

        train_test["Leak_3"] = train_test["Response"].shift(1)
        train_test["Leak_4"] = train_test["Response"].shift(-1)

        train_test = dt.Frame(train_test.drop("Response", axis=1))
        train_test.key = "Id"

        X_train = X_train[:, :, dt.join(train_test)]
        X_test = X_test[:, :, dt.join(train_test)]

        return {"bosch_train_leak": X_train, "bosch_test_leak": X_test}
