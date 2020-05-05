"""Products together 3 or more numeric features"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class ProductTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=3, max_cols=4, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        df = X.to_pandas()
        df['_value_to_return'] = df.product(axis=1, skipna=True)

        return df['_value_to_return']
