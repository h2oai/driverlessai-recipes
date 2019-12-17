"""CatBoost-style target encoding. See https://youtu.be/d6UMEmeXB6o?t=818 for short explanation"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ToDo: Completely replace pandas with datatable
class ExpandingMean(CustomTransformer):
    _multiclass = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._group_means = None
        self.dataset_mean = np.nan

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=8, relative_importance=1)

    @property
    def display_name(self):
        return "ExpandingMean"

    def transform(self, X: dt.Frame):
        transformed_X = X[:, :, dt.join(self._group_means)][:, -1]
        return dt.Frame(transformed_X.to_pandas().fillna(self.dataset_mean))

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        target = '__target__'
        X[:, target] = dt.Frame(y)
        target_is_numeric = X[:, target][:, [bool, int, float]].shape[1] > 0
        if not target_is_numeric:
            X[:, target] = dt.Frame(LabelEncoder().fit_transform(X[:, target].to_pandas().iloc[:, 0].values).ravel())

        self._group_means = X[:, dt.mean(dt.f[target]), dt.by(*self.input_feature_names)]
        self._group_means.key = self.input_feature_names
        self.dataset_mean = X[target].mean().to_numpy().ravel()[0]

        # Expanding mean transform
        X_ = X.to_pandas()[self.input_feature_names + [target]]
        X_["index"] = X_.index
        X_shuffled = X_.sample(n=len(X_), replace=False)
        X_shuffled["cnt"] = 1
        X_shuffled["cumsum"] = (X_shuffled
                                .groupby(self.input_feature_names, sort=False)['__target__']
                                .apply(lambda x: x.shift().cumsum()))
        X_shuffled["cumcnt"] = (X_shuffled
                                .groupby(self.input_feature_names, sort=False)['cnt']
                                .apply(lambda x: x.shift().cumsum()))
        X_shuffled["encoded"] = X_shuffled["cumsum"] / X_shuffled["cumcnt"]
        X_shuffled["encoded"] = X_shuffled["encoded"].fillna(self.dataset_mean)
        X_transformed = X_shuffled.sort_values("index")["encoded"].values
        return dt.Frame(X_transformed)
