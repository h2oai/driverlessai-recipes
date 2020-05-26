"""Data recipe to prepare data for Kaggle IEEE-CIS Fraud Detection https://www.kaggle.com/c/ieee-fraud-detection"""

"""
Settings for Driverless AI:

1. Update folder_path below before uploading under 'ADD DATASET' -> 'UPLOAD DATA RECIPE'

2. Start experiment and select 'IEEE.train' as training dataset, 'IEEE.test' as test dataset

3. Select 'isFraud' as 'TARGET COLUMN'

4. Select 'fold_column' as 'FOLD COLUMN'

5. Go to expert settings and click on 'LOAD CUSTOM RECIPE FROM URL', enter https://raw.githubusercontent.com/h2oai/driverlessai-recipes/rel-1.8.1/models/algorithms/catboost.py to upload CatBoost model recipe

6. On same page, switch from 'AUTO' to 'KAGGLE' under 'PIPELINE BUILDING RECIPE'

7. Go the 'FEATURES' tab in expert settings, enter "Trans_D1_start", "addr1", "card1" under 'Features to group by'

8. Click 'SAVE'

9. Set experiment dials to 7/10/1

10. Set scorer to 'AUC'

11. Start experiment
"""

import datetime
import datatable as dt
import numpy as np
import os

from h2oaicore.data import CustomData


class MyData(CustomData):
    @staticmethod
    def create_data():
        folder_path = '/home/arno/kaggle/ieee/input'  # Modify as needed

        train_identity_file = os.path.join(folder_path, 'train_identity.csv')
        test_identity_file = os.path.join(folder_path, 'test_identity.csv')
        train_transaction_file = os.path.join(folder_path, 'train_transaction.csv')
        test_transaction_file = os.path.join(folder_path, 'test_transaction.csv')
        if not (os.path.isfile(train_identity_file) and os.path.isfile(test_identity_file) and os.path.isfile(
                train_transaction_file) and os.path.isfile(test_transaction_file)):
            return []

        train_identity = dt.fread(train_identity_file)
        test_identity = dt.fread(test_identity_file)
        train_transaction = dt.fread(train_transaction_file)
        test_transaction = dt.fread(test_transaction_file)

        target = 'isFraud'
        train_identity.key = 'TransactionID'
        test_identity.key = 'TransactionID'

        # Join identity into transactions
        train = train_transaction[:, :, dt.join(train_identity)]
        test = test_transaction[:, :, dt.join(test_identity)]

        # Combine train and test for further processing
        X = dt.rbind([train, test], force=True)

        # Turn integer time column into datetime string with proper format
        startdate = datetime.datetime.strptime('2017-11-30', "%Y-%m-%d")
        pd_time = X[:, 'TransactionDT'].to_pandas()['TransactionDT'].apply(
            lambda x: (startdate + datetime.timedelta(seconds=x))
        )
        X[:, 'TransactionDT_str'] = dt.Frame(pd_time.apply(
            lambda x: datetime.datetime.strftime(x, "%Y-%m-%d %H:%M:%S"))
        )
        # Month - to be used as fold column (that way get cross-validation without shuffling future/past too much, minimize overlap between folds)
        fold_column = 'fold_column'
        X[:, fold_column] = dt.Frame(pd_time.dt.month + (pd_time.dt.year - 2017) * 12)

        # Create start times (in secs) for Dx features (which are growing linearly over time)
        for i in range(1, 16):
            X[:, 'Trans_D%d_start' % i] = dt.Frame(
                np.floor(X[:, 'TransactionDT'].to_numpy().ravel() / (24 * 60 * 60)) - X[:,
                                                                                      'D%d' % i].to_numpy().ravel())

        # re-order names
        first_names = [target, fold_column]
        names = first_names + [x for x in X.names if x not in first_names]
        X = X[:, names]

        # Split back into train and test
        train = X[:train_transaction.nrows, :]
        test = X[train_transaction.nrows:, :]
        return {'IEEE.train': train, 'IEEE.test': test}
