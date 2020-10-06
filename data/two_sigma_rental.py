from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.systemutils import user_dir
import uuid
import importlib

subprocess = importlib.import_module('sub' + 'process')


kaggle_username = "ogrellier"
kaggle_key = "12c1536d3100077e0c91034a24bb3cc8"


class TwoSigmaRental(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        import os
        from h2oaicore.systemutils_more import download
        from h2oaicore.systemutils import config

        os.putenv("KAGGLE_USERNAME", kaggle_username)
        os.putenv("KAGGLE_KEY", kaggle_key)

        # find sample submission file
        temp_path = os.path.join(user_dir(), config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)
        sub_file_dir = os.path.join(temp_path, "kaggle_%s" % str(uuid.uuid4())[:4])

        cmd_train = f'kaggle competitions download ' \
                    f'-c two-sigma-connect-rental-listing-inquiries ' \
                    f'-f train.json.zip ' \
                    f'-p {sub_file_dir} -q'
        cmd_test = f'kaggle competitions download ' \
                   f'-c two-sigma-connect-rental-listing-inquiries ' \
                   f'-f test.json.zip ' \
                   f'-p {sub_file_dir} -q'

        try:
            subprocess.check_output(cmd_train.split(), timeout=120).decode("utf-8")
        except TimeoutError:
            raise TimeoutError("Took longer than %s seconds, increase timeout")

        try:
            subprocess.check_output(cmd_test.split(), timeout=120).decode("utf-8")
        except TimeoutError:
            raise TimeoutError("Took longer than %s seconds, increase timeout")

        train = pd.read_json(os.path.join(sub_file_dir, 'train.json.zip'))
        test = pd.read_json(os.path.join(sub_file_dir, 'test.json.zip'))

        for df in [train, test]:
            df['str_features'] = df['features'].apply(lambda x: ' . '.join(x))
            df['nb_features'] = df['features'].apply(len)
            df['nb_photos'] = df['photos'].apply(len)
            df['cat_address'] = df['street_address'] + ' ' + df['display_address']

        features = [
            'bathrooms', 'bedrooms', 'building_id', 'created', 'description',
            'display_address', 'latitude', 'listing_id', 'longitude',
            'manager_id', 'price', 'street_address',
            'str_features', 'nb_features', 'nb_photos', 'cat_address'
        ]

        return {'two_sigma_train': dt.Frame(train[features+['interest_level']]),
                'two_sigma_test': dt.Frame(test[features])}
