"""Example of a transformer that operates on the entire original frame, and hence on any column(s) desired."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MySpecificColumnTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1, num_default_instances=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        col_names = X.names
        if len(col_names) >= 3:
            col_names_to_pick = [col_names[0], col_names[2]]  # can provide actual strings
        else:
            col_names_to_pick = [col_names[0]]
        X = X[:, col_names_to_pick]
        return X.to_pandas().astype(str).iloc[:, 0].str.len()
