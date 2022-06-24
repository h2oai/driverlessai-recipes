"""Unsupervised Aggregator algorithm (by Leland Wilkinson) to segment data into user-given number of exemplars"""

import datatable as dt
from datatable.models import aggregate
import numpy as np
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer


class AggregatorTransformer(CustomUnsupervisedTransformer):
    def __init__(self, n_exemplars, **kwargs):
        super().__init__(**kwargs)
        self.n_exemplars = n_exemplars

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols=1, max_cols="all", relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"n_exemplars": [250]}  # CUSTOMIZE

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        """
        Returns a column containing the exemplar row ID (0 based) for each row
        """
        agg, mapping = aggregate(dt.Frame(X),
                                 min_rows=self.n_exemplars,
                                 nd_max_bins=self.n_exemplars)
        return mapping


class AggregatorModel(CustomUnsupervisedModel):
    _included_pretransformers = ['StdFreqPreTransformer']
    _included_transformers = ['AggregatorTransformer']
    _included_scorers = ['UnsupervisedScorer']
