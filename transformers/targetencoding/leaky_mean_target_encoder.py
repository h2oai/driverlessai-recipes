"""Example implementation of a out-of-fold target encoder (leaky, not recommended)"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder


class MyLeakyCategoricalGroupMeanTargetEncoder(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _multiclass = False
    _unsupervised = False  # uses target
    _uses_target = True  # uses target
    _target_encoding_based = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._group_means = None

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=8, relative_importance=1)

    @property
    def display_name(self):
        return "MyLeakyMeanTargetGroupedBy%s" % ":".join(self.input_feature_names)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        target = '__internal_target__'
        X[:, target] = dt.Frame(y)
        target_is_numeric = X[:, target][:, [bool, int, float]].shape[1] > 0
        if target_is_numeric:
            self._group_means = X[:, dt.mean(dt.f[target]), dt.by(*self.input_feature_names)]
        else:
            X[:, target] = dt.Frame(LabelEncoder().fit_transform(X[:, target].to_pandas().iloc[:, 0].values).ravel())
            self._group_means = X[:, dt.median(dt.f[target]), dt.by(*self.input_feature_names)]
        del X[:, target]
        self._group_means.key = self.input_feature_names
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, :, dt.join(self._group_means)][:, -1]
