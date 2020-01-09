""" 
Data Recipe to perform KMeans Clustering on a dataset. 

authored by @goldentom42 (Olivier Grellier)
modified by @shivam5992  (Shivam Bansal)

__version__ = 0.1

Note:

1. This recipe can be applied on any dataset, applies Kmeans clustering and creates 
   and extra column in the dataset which is the cluster_id of each row in the dataset.

2. Currently it only uses numeric columns and ignores the categorical/text columns.

3. Users can define user inputs (global variables) for this recipe. They are following: 
   number of clusters, columns to ignore, columns to include. (see section : Optional User Inputs)

"""

from typing import Union, List
from h2oaicore.systemutils import config
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_X_y
from sklearn.utils import safe_indexing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder

""" 
    "" Optional User Inputs "" 

    | variable        | type | default | description                                             |
    | ignore_columns  | list | None    | Columns to ignore in the model, example : ID columns    |
    |                                    if not specified anything, all columns are considered   | 
    | include_columns | list | None    | Columns to specifically include in the clustering model | 
    |                                    if not specified anything, all columns are considered   |
    | num_clusters    | int  | 0       | Number of clusters to generate in the output            | 
                                         if not specified, the optimal number is obtained        | 

"""
ignore_columns = None  # Example ['ID', 'default payment next month']
include_columns = None # Example ["Age", "Gender"]
num_clusters = 0       # Example : 2,3,4 ...  (set it between 2 and 30)



# Global Settings
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


# Rewrite of sklearn metric
def my_davies_bouldin_score(X, labels):
    """Computes the Davies-Bouldin score.
    The score is defined as the ratio of within-cluster distances to
    between-cluster distances.
    Read more in the :ref:`User Guide <davies-bouldin_index>`.
    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.
    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    References
    ----------
    .. [1] Davies, David L.; Bouldin, Donald W. (1979).
       `"A Cluster Separation Measure"
       <https://ieeexplore.ieee.org/document/4766909>`__.
       IEEE Transactions on Pattern Analysis and Machine Intelligence.
       PAMI-1 (2): 224-227
    """
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=np.float)
    for k in range(n_labels):
        cluster_k = safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(
            cluster_k, [centroid]))

    # centroid_distances will contain zeros in the diagonal
    centroid_distances = pairwise_distances(centroids)

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    # Compute score avoiding division by zero by adding an epsilon
    # this leads to high values in the diagonal's result
    score = (intra_dists[:, None] + intra_dists) / (centroid_distances + 1e-15)

    # Simply put the diagonal to zero
    score[np.eye(centroid_distances.shape[0]) == 1] = 0

    # Here is the original code
    # score = (intra_dists[:, None] + intra_dists) / (centroid_distances)
    # score[score == np.inf] = np.nan
    return np.mean(np.nanmax(score, axis=1))


class KMeansClustering(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        if X is None:
            return []


        if include_columns != None:
            for _ in include_columns:
                if _ not in list(X.names):
                    raise ValueError(str(_) + " is not present in the dataset")

            features = [_ for _ in include_columns if _ in list(X.names)]:
        else:
            features = [_f for _f in X.names if _f not in ignore_columns]

        if len(features) = 0:
            raise ValueError("Unable to cluster: No Features Selected")

        # Restrict features to numerical for now
        X_clust = X[:, features][:, [int, float]].to_numpy()

        # Apply min max scaling
        X_clust = MinMaxScaler().fit_transform(X_clust)

        # Go trhough possible numbers of clusters
        best_score = None
        best_n_clust = None
        best_clust_ids = None

        ## if number of clusters is pre-defined by user, then dont find the optimal
        if num_clusters > 1:
            n_clusters = num_clusters
            model = KMeans(n_clusters=n_clusters, n_jobs=NUM_JOBS).fit(X_clust)
            clust_ids = model.predict(X_clust)
            score = my_davies_bouldin_score(
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
                score = my_davies_bouldin_score(
                    X_clust,
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
