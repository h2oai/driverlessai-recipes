"""Outlier detection with Local Outlier Factor"""
import copy
from typing import List

from h2oaicore.systemutils import IgnoreEntirelyError, update_precision
from sklearn.neighbors import LocalOutlierFactor

"""
https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection-with-local-outlier-factor
"""
import datatable as dt
import numpy as np
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer


class LocalOutlierFactorTransformer(CustomUnsupervisedTransformer):
    _can_use_gpu = False
    _parallel_task = True  # if enabled, fit_transform and transform will be given self.n_jobs and kwargs['n_jobs']
    # n_jobs will be  >= 1 (adaptive to system resources and tasks), otherwise 1 if _parallel_task = False

    def __init__(self,
                 num_cols: List[str] = list(),
                 output_features_to_drop=list(),
                 n_neighbors=20,
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2,
                 novelty=False,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        init_args_dict = locals().copy()
        self.params = {k: v for k, v in init_args_dict.items() if k in self.get_parameter_choices()}
        self._output_features_to_drop = output_features_to_drop

    @staticmethod
    def get_parameter_choices():
        """
        Possible parameters to use as mutations, where first value is default value
        See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        :return:
        """

        return dict(n_neighbors=[20],  # could add to list other values
                    algorithm=['auto'],  # could add to list 'ball_tree', 'kd_tree', 'brute'
                    leaf_size=[30],  # could add to list other values
                    metric=['minkowski'],  # could add [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
                    p=[2],  # could add p=1 i.e. manhattan_distance or p=2 for euclidean_distance
                    novelty=[False],  # could add True
                    )

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols="all")

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        if X.nrows <= 2:
            raise IgnoreEntirelyError
        params = copy.deepcopy(self.params)
        params.update(dict(n_jobs=self.n_jobs))
        print("LocalOutlierFactorTransformer params: %s" % params)
        self.model = LocalOutlierFactor(**params)
        # make float, replace of nan/inf won't work on int
        X = update_precision(X, fixup_almost_numeric=False)
        X.replace([None, np.nan, np.inf, -np.inf], 0.0)
        X = X.to_numpy()
        return self.model.fit_predict(X)

    def transform(self, X: dt.Frame, y: np.array = None):
        # no state, always finds outliers in any given dataset
        return self.fit_transform(X)


class LocalOutlierFactorModel(CustomUnsupervisedModel):
    _included_pretransformers = ['OrigFreqPreTransformer']  # frequency-encode categoricals, keep numerics as is
    _included_transformers = ["LocalOutlierFactorTransformer"]
    _included_scorers = ['UnsupervisedScorer']  # trivial, nothing to score
