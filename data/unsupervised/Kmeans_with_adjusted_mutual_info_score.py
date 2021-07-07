"""Kmeans Clustering with Adjusted Mutual Info Score to evaluate cluster quality."""

"""
__version__ = 0.1

Note:
Users can define (optional) user inputs: number of clusters, columns to ignore, columns to include. 
(see section : Optional User Inputs)

"""

from typing import Union, List
from h2oaicore.systemutils import config
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_mutual_info_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

""" 
	"" Optional User Inputs "" 

	| variable        | type | default | description                                             |
	| ignore_columns  | list | []      | Columns to ignore in the model, example : ID columns    |
	|                                    if not specified, all columns are considered            | 
	| include_columns | list | []      | Columns to specifically include in the clustering model | 
	|                                    if not specified, all columns are considered            |
	| num_clusters    | int  | 0       | Number of clusters to generate in the output            | 
										 if not specified, optimal number is obtained            | 

"""
ignore_columns = []  # <list> default = [], if default then uses all columns
include_columns = []  # <list> default = [], if default then uses all columns
num_clusters = 0  # <int>  set 0 to optimize the number of clusters automatically
target = 'default payment next month'

# Global Settings for optimizing number of clusters
MIN_CLUSTERS = 3
MAX_CLUSTERS = 15
CLUSTER_STEP_SIZE = 1
NUM_JOBS = 2


def check_number_of_labels(n_labels, n_samples):
    """Check that number of labels are valid.
    Parameters
    ----------
    n_labels : int
        Number of labels
    n_samples : int
        Number of samples
    """
    if not 1 < n_labels < n_samples:
        raise ValueError("Number of labels is %d. Valid values are 2 "
                         "to n_samples - 1 (inclusive)" % n_labels)

def get_score(X, labels):
    score = adjusted_mutual_info_score(X, labels)
    return score


class KMeansClustering(CustomData):
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
        if not isinstance(num_clusters, int):
            raise ValueError("Column: 'num_clusters' should be <int>")

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

        # Go through possible numbers of clusters
        best_score = None
        best_n_clust = None
        best_clust_ids = None

        ## if number of clusters is pre-defined by user, then dont find the optimal
        if num_clusters > 1:
            model = KMeans(n_clusters=num_clusters, n_jobs=NUM_JOBS).fit(X_clust)
            clust_ids = model.predict(X_clust)
            score = get_score(
                X_clust,
                clust_ids
            )
            best_score = score
            best_n_clust = num_clusters
            best_clust_ids = clust_ids

        else:
            for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS, CLUSTER_STEP_SIZE):
                model = KMeans(n_clusters=n_clusters, n_jobs=NUM_JOBS).fit(X_clust)
                clust_ids = model.predict(X_clust)

                X1 = X.to_pandas()
                score = get_score(
                    list(X1[target]),
                    clust_ids
                )
                improve = False
                if best_score is None:
                    improve = True
                elif best_score > score:
                    improve = True

                if improve:
                    best_score = score
                    best_n_clust = n_clusters
                    best_clust_ids = clust_ids

        if best_score is None:
            return []
        else:
            X[:, f'cluster_ids_{best_n_clust}'] = dt.Frame(best_clust_ids)
        return X