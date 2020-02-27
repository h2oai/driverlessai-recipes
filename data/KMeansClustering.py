"""Data Recipe to perform KMeans Clustering on a dataset."""

"""
__version__ = 0.1

authored by @goldentom42 (Olivier Grellier)
modified by @shivam5992  (Shivam Bansal)


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
	| ignore_columns  | list | []      | Columns to ignore in the model, example : ID columns    |
	|                                    if not specified, all columns are considered            | 
	| include_columns | list | []      | Columns to specifically include in the clustering model | 
	|                                    if not specified, all columns are considered            |
	| num_clusters    | int  | 0       | Number of clusters to generate in the output            | 
										 if not specified, optimal number is obtained            | 

"""
ignore_columns  = []    # <list> default = [], if default then uses all columns
include_columns = []    # <list> default = [], if default then uses all columns
num_clusters    = 3     # <int>  set 0 to optimize the number of clusters automatically


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
					raise ValueError("Column: '"+str(_) + "' is not present in the dataset")
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
			if miss_percent >= 0.3: # ignore columns having more than 30% missing values
				ignore_.append(col)
			elif miss_percent > 0.0: # impute by mean for other columns with missing values
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
