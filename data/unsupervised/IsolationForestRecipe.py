"""Data Recipe to perform Isolation Forest Clustering on a dataset."""

"""
__version__ = 0.1

Note:

Users can define (optional) user inputs: number of clusters, columns to ignore, columns to include. 
(see section : Optional User Inputs)

"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import IsolationForest

""" 
	"" Optional User Inputs "" 

	| variable        | type | default | description                                             |
	| ignore_columns  | list | []      | Columns to ignore in the model, example : ID columns    |
	|                                    if not specified, all columns are considered            | 
	| include_columns | list | []      | Columns to specifically include in the clustering model | 
	|                                    if not specified, all columns are considered            | 

"""
ignore_columns = []  # <list> default = [], if default then uses all columns
include_columns = []  # <list> default = [], if default then uses all columns

# Global Settings for optimizing number of clusters
MIN_CLUSTERS = 3
MAX_CLUSTERS = 15
CLUSTER_STEP_SIZE = 1
NUM_JOBS = 2


class IsolationForestClustering(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        if X is None:
            return []

        # check the datatype of user-defined columns
        if not isinstance(include_columns, list):
            raise ValueError("Variable: 'include_columns' should be <list>")
        if not isinstance(ignore_columns, list):
            raise ValueError("Column: 'ignore_columns' should be <list>")
        # if not isinstance(num_clusters, int):
        # raise ValueError("Column: 'num_clusters' should be <int>")

        ## validate user-inputs and override the columns given by user
        features = list(X.names)
        if len(include_columns) > 0:
            for _ in include_columns:
                if _ not in list(X.names):
                    raise ValueError("Column: '" + str(_) + "' is not present in the dataset")
            features = include_columns

        ## list to ignore specific columns given by user
        features = [_f for _f in features if _f not in ignore_columns]

        ## handle columns with missing values 
        ignore_ = []
        X_df = X.to_pandas()
        for col in features:
            # label encode categorical columns
            # refer - https://github.com/h2oai/driverlessai-recipes/pull/68#discussion_r365133392

            if X_df[col].dtype == "object":
                X_df[f"{col}_enc"] = LabelEncoder().fit_transform(X_df[col].to_numpy())
                ignore_.append(col)

            miss_percent = X_df[col].isna().sum() / X_df.shape[0]
            if miss_percent >= 0.3:  # ignore columns having more than 30% missing values
                ignore_.append(col)
            elif miss_percent > 0.0:  # impute by mean for other columns with missing values
                X_df[col] = X_df[col].fillna(X_df[col].mean())

        features = [f for f in features if f not in ignore_]
        features += [_f for _f in X_df.columns if "_enc" in _f]
        if len(features) == 0:
            raise ValueError("Unable to cluster: No useful features available")

        X_clust = X_df[features].values

        # Apply min max scaling
        X_clust = MinMaxScaler().fit_transform(X_clust)

        clf = IsolationForest(random_state=0).fit(X_clust)
        X['isolationforest_clusters'] = clf.predict(X_clust)

        return X
