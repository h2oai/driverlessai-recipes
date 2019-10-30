"""Create airlines dataset"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd


class CatChallenge(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None):

        # to be usd with models.algorithms.logistic_regression.py with
        # _kaggle = True
        # _kaggle_features = True
        # _kaggle_mode = True
        train = dt.fread("/home/jon/kaggle/cat/inputs/train.csv.zip")
        train_orig = dt.Frame(train)
        train['sample_weight'] = dt.Frame(np.array([1.0] * train.shape[0]))
        test = dt.fread("/home/jon/kaggle/cat/inputs/test.csv.zip")
        test_orig = dt.Frame(test)
        test['sample_weight'] = dt.Frame(np.array([0.0] * test.shape[0]))
        test['target'] = dt.Frame(np.array([0.0] * test.shape[0]))
        final = dt.rbind([train, test])

        return {'catmerged': final, 'cattrain': train_orig, 'cattest': test_orig}
