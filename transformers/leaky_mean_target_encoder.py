from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config
import datatable as dt
import numpy as np
import h2o
import uuid
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


class MyLeakyCategoricalGroupMeanTargetEncoder(CustomTransformer):
    _multiclass = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._group_means = None

    @staticmethod
    def get_default_properties(*args, **kwargs):
        return dict(col_type="categorical", min_cols=1, max_cols=8, relative_importance=1)

    @property
    def display_name(self):
        return "MyLeakyMeanTargetGroupedBy%s" % ":".join(self.input_feature_names)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        target = '__internal_target__'
        X[:, target] = dt.Frame(y)
        self._group_means = X[:, dt.mean(dt.f[target]), dt.by(*self.input_feature_names)]
        del X[:, target]
        self._group_means.key = self.input_feature_names
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, :, dt.join(self._group_means)][:, -1]
