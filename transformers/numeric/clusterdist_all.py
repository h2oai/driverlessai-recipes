"""Cluster Distance for all columns"""

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
from typing import List

from h2oaicore.transformers_cuml import ClusterDistCUMLTransformer


class ClusterDistCUMLTransformerAll(ClusterDistCUMLTransformer, CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(self, num_cols: List[str] = None, n_clusters=20, max_iter=50, tol=1e-2, init='scalable-k-means++',
                 oversampling_factor=1, max_samples_per_batch=32768,
                 output_features_to_drop=list(), copy=False, **kwargs):
        super().__init__(num_cols=num_cols, n_components=n_clusters, max_iter=max_iter, tol=tol,
                         init=init, oversampling_factor=oversampling_factor,
                         max_samples_per_batch=max_samples_per_batch,
                         output_features_to_drop=output_features_to_drop, copy=copy, **kwargs)

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=2, max_cols="all", relative_importance=1)

    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)

    def transform(self, X, y=None, **fit_params):
        return super().transform(X, y, **fit_params)

    from h2oaicore.mojo import MojoWriter, MojoFrame

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        return super().to_mojo(mojo, iframe, group_uuid, group_name)
