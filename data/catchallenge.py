"""Create cat challenge dataset"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
import os


class CatChallenge(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None):
        # to be used with models.algorithms.logistic_regression.py with
        # _kaggle = True
        # _kaggle_features = True
        # _kaggle_mode = True
        path = "/home/jon/kaggle/cat/inputs/"
        if not os.path.exists(path):
            return []

        train = dt.fread(os.path.join(path, "train.csv.zip"))
        train_orig = dt.Frame(train)
        train['sample_weight'] = dt.Frame(np.array([1.0] * train.shape[0]))
        test = dt.fread(os.path.join(path, "test.csv.zip"))
        test_orig = dt.Frame(test)
        test['sample_weight'] = dt.Frame(np.array([1.0] * test.shape[0]))
        test['target'] = dt.Frame(np.array([0] * test.shape[0], dtype=int))
        final = dt.rbind([train, test])

        return {'catmerged': final, 'cattrain': train_orig, 'cattest': test_orig}
