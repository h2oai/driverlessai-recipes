"""Truncated SVD for all columns"""

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
from h2oaicore.transformers import TruncSVDNumTransformer
from typing import List


class TruncatedSvdNumAll(TruncSVDNumTransformer, CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(self, num_cols: List[str] = None, n_components=3, random_state=42, algorithm='randomized',
                 output_features_to_drop=list(), copy=False, **kwargs):
        super().__init__(num_cols=num_cols, n_components=n_components, random_state=random_state, algorith=algorithm,
                         output_features_to_drop=output_features_to_drop, copy=copy, **kwargs)

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=2, max_cols="all", relative_importance=1)

    def fit_transform(self, X, y=None, **fit_params):
        if isinstance(X, dt.Frame):
            X = X.to_pandas()
        return super().fit_transform(X, y, **fit_params)

    def transform(self, X, y=None, **fit_params):
        if isinstance(X, dt.Frame):
            X = X.to_pandas()
        return super().transform(X, y, **fit_params)

    from h2oaicore.mojo import MojoWriter, MojoFrame

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        return super().to_mojo(mojo, iframe, group_uuid, group_name)
