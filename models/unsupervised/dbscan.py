"""Clustering using DBScan"""
import datatable as dt
import numpy as np
import sklearn as sk
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer


class DBScanTransformer(CustomUnsupervisedTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols="all")

    @staticmethod
    def get_parameter_choices():
        return dict(eps=[0.1, 0.5, 1, 2, 5], leaf_size=[1, 5, 10, 25, 100])

    def __init__(self, eps=None, leaf_size=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.leaf_size = leaf_size

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.model = sk.cluster.DBSCAN(eps=self.eps, min_samples=2, leaf_size=self.leaf_size)
        X = X.to_pandas().fillna(0)
        return self.model.fit_predict(X)

    def transform(self, X: dt.Frame, y: np.array = None):
        return self.fit_transform(X)


class DBScanModel(CustomUnsupervisedModel):
    _included_pretransformers = ['StdFreqPreTransformer']  # standardize numericals, frequency encode categoricals

    _included_transformers = ["DBScanTransformer"]

    _included_scorers = ['SilhouetteScorer', 'CalinskiHarabaszScorer', 'DaviesBouldinScorer']  # from DAI built-in
