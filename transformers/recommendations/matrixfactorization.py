"""Collaborative filtering features using various techniques of Matrix Factorization for recommendations.
Recommended for large data"""

"""
Add the user column name and item column name in recipe_dict in config to match the
column names as per the dataset or use the default 'user' and 'item' respectively in your dataset

Sample Datasets
# Netflix - https://www.kaggle.com/netflix-inc/netflix-prize-data
recipe_dict = "{'user_col': 'user', 'item_col': 'movie'}"

# MovieLens - https://grouplens.org/datasets/movielens/
recipe_dict = "{'user_col': 'userId', 'item_col': 'movieId'}"

# RPackages - https://www.kaggle.com/c/R/data
recipe_dict = "{'user_col': 'User', 'item_col': 'Package'}"
"""

import datatable as dt
import numpy as np
import pandas as pd

import h2o4gpu
import scipy

from h2oaicore.systemutils import config
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.separators import extra_prefix, orig_feat_prefix, col_sep
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


class RecH2OMFTransformer(CustomTransformer):
    _allow_transform_to_modify_output_feature_names = True
    _multiclass = False
    _can_use_gpu = True
    _mf_type = "h2o4gpu"

    def __init__(self, n_components=50, _lambda=0.1, batches=1, max_iter=100, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.user_col = config.recipe_dict['user_col'] if "user_col" in config.recipe_dict else "user"
        self.item_col = config.recipe_dict['item_col'] if "item_col" in config.recipe_dict else "item"

        if self.__class__._mf_type == "h2o4gpu":
            self._n_components = n_components
            self._lambda = _lambda
            self._batches = batches
            self._max_iter = max_iter
        elif self.__class__._mf_type == "nmf":
            self._n_components = n_components
            self._alpha = alpha
            self._max_iter = max_iter

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1, num_default_instances=1)

    @staticmethod
    def get_parameter_choices():
        return {"n_components": [10, 30, 50, 70, 100],
                "_lambda": [0.01, 0.05, 0.1],
                "batches": [1],
                "max_iter": [10, 50, 100, 200],
                "alpha": [0.01, 0.05, 0.1]}

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        if len(np.unique(self.labels)) == 2:
            le = LabelEncoder()
            self.labels = le.fit_transform(self.labels)
            y = np.array(le.transform(y), dtype="float32")
        else:
            y = np.array(y, dtype="float32")

        X = X[:, [self.user_col, self.item_col]]

        self.user_le = LabelEncoder()
        self.item_le = LabelEncoder()

        X[:, self.user_col] = dt.Frame(self.user_le.fit_transform(X[:, self.user_col]))
        X[:, self.item_col] = dt.Frame(self.item_le.fit_transform(X[:, self.item_col]))

        X_pd = X.to_pandas()

        if len(np.unique(self.labels)) == 2:
            kfold = StratifiedKFold(n_splits=10)
        else:
            kfold = KFold(n_splits=10)

        preds = np.full(X.nrows, fill_value=np.nan)

        for train_index, val_index in kfold.split(X_pd, y):
            X_train, y_train = X_pd.iloc[train_index,], y[train_index]
            X_val, y_val = X_pd.iloc[val_index,], y[val_index]

            X_val2 = X_val[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (
                X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]
            y_val2 = y_val[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (
                X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]

            X_panel = pd.concat([X_train, X_val2], axis=0)

            users, user_indices = np.unique(np.array(X_panel[self.user_col], dtype="int32"), return_inverse=True)
            items, item_indices = np.unique(np.array(X_panel[self.item_col], dtype="int32"), return_inverse=True)

            X_train_user_item_matrix = scipy.sparse.coo_matrix(
                (y_train, (user_indices[:len(X_train)], item_indices[:len(X_train)])), shape=(len(users), len(items)))
            X_train_shape = X_train_user_item_matrix.shape

            X_val_user_item_matrix = scipy.sparse.coo_matrix(
                (np.ones(len(X_val2), dtype="float32"), (user_indices[len(X_train):], item_indices[len(X_train):])),
                shape=X_train_shape)

            if self.__class__._mf_type == "h2o4gpu":
                factorization = h2o4gpu.solvers.FactorizationH2O(self._n_components, self._lambda,
                                                                 max_iter=self._max_iter)
                factorization.fit(X_train_user_item_matrix, X_BATCHES=self._batches, THETA_BATCHES=self._batches)
                preds[val_index[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (
                    X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]] = factorization.predict(
                    X_val_user_item_matrix).data
            elif self.__class__._mf_type == "nmf":
                factorization = NMF(n_components=self._n_components, alpha=self._alpha, max_iter=self._max_iter)
                user_matrix = factorization.fit_transform(X_train_user_item_matrix)
                item_matrix = factorization.components_.T
                val_users = np.take(user_matrix, X_val_user_item_matrix.row, axis=0)
                val_items = np.take(item_matrix, X_val_user_item_matrix.col, axis=0)
                preds[val_index[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (
                    X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]] = np.sum(val_users * val_items,
                                                                                             axis=1)

        users, user_indices = np.unique(np.array(X_pd[self.user_col], dtype="int32"), return_inverse=True)
        items, item_indices = np.unique(np.array(X_pd[self.item_col], dtype="int32"), return_inverse=True)

        X_train_user_item_matrix = scipy.sparse.coo_matrix(
            (y_train, (user_indices[:len(X_train)], item_indices[:len(X_train)])), shape=(len(users), len(items)))
        self.X_train_shape = X_train_user_item_matrix.shape

        if self.__class__._mf_type == "h2o4gpu":
            self.factorization = h2o4gpu.solvers.FactorizationH2O(self._n_components, self._lambda,
                                                                  max_iter=self._max_iter)
            self.factorization.fit(X_train_user_item_matrix, X_BATCHES=self._batches, THETA_BATCHES=self._batches)
        elif self.__class__._mf_type == "nmf":
            factorization = NMF(n_components=self._n_components, alpha=self._alpha, max_iter=self._max_iter)
            self.user_matrix = factorization.fit_transform(X_train_user_item_matrix)
            self.item_matrix = factorization.components_.T

        # output feature names
        if self.__class__._mf_type == "h2o4gpu":
            self._output_feature_names = [(f"{self.display_name}{orig_feat_prefix}{self.user_col}{col_sep}"
                                          f"{self.item_col}.n_components={self._n_components},"
                                          f"lambda={self._lambda},batches={self._batches},max_iter={self._max_iter}")]
        elif self.__class__._mf_type == "nmf":
            self._output_feature_names = [(f"{self.display_name}{orig_feat_prefix}{self.user_col}{col_sep}"
                                          f"{self.item_col}.n_components={self._n_components},"
                                          f"alpha={self._alpha},max_iter={self._max_iter}")]
        # output feature descriptions
        self._feature_desc = [f"Recommender transformer ({self.__class__._mf_type}): " + self._output_feature_names[0]]

        return preds

    def transform(self, X: dt.Frame):
        X = X[:, [self.user_col, self.item_col]]

        preds = np.full(X.nrows, fill_value=np.nan)

        X_pd = X.to_pandas()

        X_test = X_pd[
            (X_pd[self.user_col].isin(self.user_le.classes_)) & (X_pd[self.item_col].isin(self.item_le.classes_))]
        X_test[self.user_col] = self.user_le.transform(X_test[self.user_col])
        X_test[self.item_col] = self.item_le.transform(X_test[self.item_col])

        X_test_user_item_matrix = scipy.sparse.coo_matrix(
            (np.ones(len(X_test), dtype="float32"), (X_test[self.user_col], X_test[self.item_col])),
            shape=self.X_train_shape)

        if self.__class__._mf_type == "h2o4gpu":
            preds[(X_pd[self.user_col].isin(self.user_le.classes_)) & (
                X_pd[self.item_col].isin(self.item_le.classes_))] = self.factorization.predict(
                X_test_user_item_matrix).data
        elif self.__class__._mf_type == "nmf":
            test_users = np.take(self.user_matrix, X_test_user_item_matrix.row, axis=0)
            test_items = np.take(self.item_matrix, X_test_user_item_matrix.col, axis=0)
            preds[(X_pd[self.user_col].isin(self.user_le.classes_)) & (
                X_pd[self.item_col].isin(self.item_le.classes_))] = np.sum(test_users * test_items, axis=1)

        return preds


class RecNMFTransformer(RecH2OMFTransformer):
    _can_use_gpu = False
    _mf_type = "nmf"
