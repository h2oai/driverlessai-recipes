"""Outlier detection with Local Outlier Factor"""
from h2oaicore.systemutils import IgnoreEntirelyError
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
    _parallel_task = False

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols="all")

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        if X.nrows <= 2:
            raise IgnoreEntirelyError
        self.model = LocalOutlierFactor()
        X = X.to_pandas().fillna(0)
        return self.model.fit_predict(X)

    def transform(self, X: dt.Frame, y: np.array = None):
        # no state, always finds outliers in any given dataset
        return self.fit_transform(X)


class LocalOutlierFactorModel(CustomUnsupervisedModel):
    _included_pretransformers = ['OrigFreqPreTransformer']  # frequency-encode categoricals, keep numerics as is
    _included_transformers = ["LocalOutlierFactorTransformer"]
    _included_scorers = ['UnsupervisedScorer']  # trivial, nothing to score
