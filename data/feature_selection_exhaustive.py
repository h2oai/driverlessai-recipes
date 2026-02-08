"""Exhaustive Feature Selection using an sklearn estimator."""

"""
Settings for this recipe:

TARGET_COLUMN: Column name of target variable
ESTIMATOR: Base sklearn estimator
MIN_FEATURES: Minimum number of final features to select
MAX_FEATURES: Maximum number of final features to select
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
MIN_FEATURES = 10
MAX_FEATURES = 15
SCORING = 'accuracy'
CV = 5


class ExhaustiveFeatureSelection(CustomData):
    _modules_needed_by_name = ["mlxtend==0.23.4"]

    @staticmethod
    def create_data(X: dt.Frame = None) -> pd.DataFrame:
        if X is None:
            return []

        from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

        X = X.to_pandas()
        y = X[TARGET_COLUMN].values
        X.drop(TARGET_COLUMN, axis=1, inplace=True)

        efs = EFS(ESTIMATOR,
                  min_features=MIN_FEATURES,
                  max_features=MAX_FEATURES,
                  scoring=SCORING,
                  cv=CV,
                  n_jobs=-1)

        efs.fit(X, y)

        X_fs = X.iloc[:, list(efs.best_idx_)]

        return X_fs
