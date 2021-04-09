"""Truncated SVD for all columns"""

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
from h2oaicore.transformers_more import OneHotEncodingTransformer


class OHETransformer(OneHotEncodingTransformer, CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _included_model_classes = None  # to stop GLMModel from being only model allowed

    def __init__(self, cat_cols=[], max_cat_bins=20, sort_order="lexical", multi_class=False,
                 output_features_to_drop=list(), copy=True, random_state=42, **kwargs):
        super().__init__(cat_cols=cat_cols, max_cat_bins=max_cat_bins, sort_order=sort_order, multi_class=multi_class,
                 output_features_to_drop=output_features_to_drop, copy=copy, random_state=random_state, **kwargs)

    @staticmethod
    def get_default_properties():
        return dict(col_type="ohe_categorical", min_cols=1, max_cols=1, relative_importance=1)

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
