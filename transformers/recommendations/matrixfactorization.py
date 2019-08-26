"""Collaborative filtering features using various techniques of Matrix Factorization for recommendations"""

"""
Please edit the user column name and item name in the transformer initializationto match the
column names as per the dataset or use the default 'user' and 'item' respectively in your dataset

Sample Datasets
Netflix - https://www.kaggle.com/netflix-inc/netflix-prize-data
user_col = user
item_col = movie

MovieLens - https://grouplens.org/datasets/movielens/
user_col = userId
item_col = movieId

RPackages - https://www.kaggle.com/c/R/data
user_col = User
item_col = Package
"""

import datatable as dt
import numpy as np
import pandas as pd

import scipy
import h2o4gpu

from h2oaicore.transformer_utils import CustomTransformer
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


class RecH2OMFTransformer(CustomTransformer):
    _multiclass = False
    _can_use_gpu = True
    _mf_type = "h2o4gpu"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.user_col = "user" ## edit this to the user column name
        self.item_col = "item" ## edit this to the item column name

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1, num_default_instances=1)

    def set_params(self):
        if self.__class__._mf_type == "h2o4gpu":
            self.params = {"n_components": 50,
                           "lambda": 0.01,
                           "batches": 1,
                           "max_iter": 50}
        elif self.__class__._mf_type == "nmf":
            self.params = {"n_components": 50,
                           "alpha": 0.01,
                           "max_iter": 50}

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.set_params()

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

        kfold = KFold(n_splits=5)
        preds = np.zeros(X.nrows)

        for train_index, val_index in kfold.split(X_pd, y):
            X_train, y_train = X_pd.iloc[train_index,], y[train_index]
            X_val, y_val = X_pd.iloc[val_index,], y[val_index]
            
            X_val2 = X_val[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]
            y_val2 = y_val[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]
            
            X_panel = pd.concat([X_train, X_val2], axis=0)
            
            users, user_indices = np.unique(np.array(X_panel[self.user_col], dtype="int32"), return_inverse=True)
            items, item_indices = np.unique(np.array(X_panel[self.item_col], dtype="int32"), return_inverse=True)
            
            X_train_user_item_matrix = scipy.sparse.coo_matrix((y_train, (user_indices[:len(X_train)], item_indices[:len(X_train)])), shape=(len(users), len(items)))
            X_train_shape = X_train_user_item_matrix.shape
            
            X_val_user_item_matrix = scipy.sparse.coo_matrix((np.ones(len(X_val2), dtype="float32"), (user_indices[len(X_train):], item_indices[len(X_train):])), shape=X_train_shape)
            
            if self.__class__._mf_type == "h2o4gpu":
                factorization = h2o4gpu.solvers.FactorizationH2O(self.params["n_components"], self.params["lambda"], max_iter=self.params["max_iter"])
                factorization.fit(X_train_user_item_matrix, X_BATCHES=self.params["batches"], THETA_BATCHES=self.params["batches"])
                preds[val_index[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]] = factorization.predict(X_val_user_item_matrix).data
            elif self.__class__._mf_type == "nmf":
                factorization = NMF(n_components=self.params["n_components"], alpha=self.params["alpha"], max_iter=self.params["max_iter"])
                user_matrix = factorization.fit_transform(X_train_user_item_matrix)
                item_matrix = factorization.components_.T
                val_users = np.take(user_matrix, X_val_user_item_matrix.row, axis=0)
                val_items = np.take(item_matrix, X_val_user_item_matrix.col, axis=0)
                preds[val_index[(X_val[self.user_col].isin(np.unique(X_train[self.user_col]))) & (X_val[self.item_col].isin(np.unique(X_train[self.item_col])))]] = np.sum(val_users * val_items, axis=1)
            
        users, user_indices = np.unique(np.array(X_pd[self.user_col], dtype="int32"), return_inverse=True)
        items, item_indices = np.unique(np.array(X_pd[self.item_col], dtype="int32"), return_inverse=True)

        X_train_user_item_matrix = scipy.sparse.coo_matrix((y_train, (user_indices[:len(X_train)], item_indices[:len(X_train)])), shape=(len(users), len(items)))
        self.X_train_shape = X_train_user_item_matrix.shape

        if self.__class__._mf_type == "h2o4gpu":
            self.factorization = h2o4gpu.solvers.FactorizationH2O(self.params["n_components"], self.params["lambda"], max_iter=self.params["max_iter"])
            self.factorization.fit(X_train_user_item_matrix, X_BATCHES=self.params["batches"], THETA_BATCHES=self.params["batches"])
        elif self.__class__._mf_type == "nmf":
            factorization = NMF(n_components=self.params["n_components"], alpha=self.params["alpha"], max_iter=self.params["max_iter"])
            self.user_matrix = factorization.fit_transform(X_train_user_item_matrix)
            self.item_matrix = factorization.components_.T

        return preds

    def transform(self, X: dt.Frame):
        X = X[:, [self.user_col, self.item_col]]
        
        preds = np.zeros(X.nrows)

        X_pd = X.to_pandas()
        
        X_test = X_pd[(X_pd[self.user_col].isin(self.user_le.classes_)) & (X_pd[self.item_col].isin(self.item_le.classes_))]
        X_test[self.user_col] = self.user_le.transform(X_test[self.user_col])
        X_test[self.item_col] = self.item_le.transform(X_test[self.item_col])

        X_test_user_item_matrix = scipy.sparse.coo_matrix((np.ones(len(X_test), dtype="float32"), (X_test[self.user_col], X_test[self.item_col])), shape=self.X_train_shape)

        if self.__class__._mf_type == "h2o4gpu":
            preds[(X_pd[self.user_col].isin(self.user_le.classes_)) & (X_pd[self.item_col].isin(self.item_le.classes_))] = self.factorization.predict(X_test_user_item_matrix).data
        elif self.__class__._mf_type == "nmf":
            test_users = np.take(self.user_matrix, X_test_user_item_matrix.row, axis=0)
            test_items = np.take(self.item_matrix, X_test_user_item_matrix.col, axis=0)
            preds[(X_pd[self.user_col].isin(self.user_le.classes_)) & (X_pd[self.item_col].isin(self.item_le.classes_))] = np.sum(test_users * test_items, axis=1)

        return preds

class RecNMFTransformer(RecH2OMFTransformer):
    _can_use_gpu = False
    _mf_type = "nmf"
