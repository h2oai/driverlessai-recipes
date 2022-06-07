"""Bidirectional Feature Selection using an sklearn estimator."""

"""
Settings for this recipe:

TARGET_COLUMN: Column name of target variable
ESTIMATOR: Base sklearn estimator
K_FEATURES: Number of final features to select
SCORING: Scoring metric
CV: Number of cross-validation folds

More details available here: http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector

P.S. Categorical inputs need to be converted to numeric before running feature selection.
"""

import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.data import CustomData
import typing
from sklearn.linear_model import LogisticRegression

# Please edit these before usage (default values are for credit card dataset)
TARGET_COLUMN = 'default payment next month'
ESTIMATOR = LogisticRegression()
K_FEATURES = 10
SCORING = 'accuracy'
CV = 5


class BidirectionalFeatureSelection(CustomData):
    _modules_needed_by_name = ["mlxtend"]

    @staticmethod
    def create_data(X: dt.Frame = None) -> pd.DataFrame:
        if X is None:
            return []

        from mlxtend.feature_selection import SequentialFeatureSelector as SFS

        X = X.to_pandas()
        y = X[TARGET_COLUMN].values
        X.drop(TARGET_COLUMN, axis=1, inplace=True)

        sfs = SFS(ESTIMATOR,
                  k_features=K_FEATURES,
                  forward=True,
                  floating=True,
                  scoring=SCORING,
                  cv=CV,
                  n_jobs=-1)

        sfs.fit(X, y)

        X_fs = X.iloc[:, list(sfs.k_feature_idx_)]

        return X_fs
