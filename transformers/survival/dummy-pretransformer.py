"""Dummy Pre-Transformer to use as a template for custom pre-transformer recipes.
   This transformer consumes all features at once, adds 'pre:' to the names and passes
   them down to transformer level and GA as-is."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import uuid

from h2oaicore.systemutils import temporary_files_path, config, remove
from h2o.estimators.coxph import H2OCoxProportionalHazardsEstimator
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
from h2oaicore.separators import extra_prefix, orig_feat_prefix

class DummyIdentityPreTransformer(CustomTransformer):

    # only works with binomial problem for now
    _regression = True
    _binary = True
    _multiclass = True
    _numeric_output = False
    _can_be_pretransformer = True
    _default_as_pretransformer = True
    _must_be_pretransformer = True
    _only_as_pretransformer = True

    def __init__(self, context=None, **kwargs):
        super().__init__(context=context, **kwargs)

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None, **kwargs):

        X_original = X

        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        self._output_feature_names = ["pre:" + x for x in list(X_original.names)]
        self._feature_desc = ["Pre-transformed feature " + x for x in list(X_original.names)]

        return X_original

    def transform(self, X: dt.Frame):

        return X


