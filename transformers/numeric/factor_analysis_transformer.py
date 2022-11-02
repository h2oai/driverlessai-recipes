"""Factor Analysis Transformer"""

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
from typing import List


class FactorAnalysisTransformer(CustomTransformer):
    _unsupervised = True

    _unsupervised = True
    _display_name = "Factor Analysis (FA) Transformer"

    @staticmethod
    def get_default_properties():
        return dict(
            col_type="numeric", min_cols=2, max_cols="all", relative_importance=1
        )

    @staticmethod
    def get_parameter_choices():
        return dict(n_components=[1, 2, 3])

    def __init__(self, n_components=1, **kwargs):
        super().__init__(**kwargs)
        self._n_components = n_components

    def fit_transform(self, X, y=None, **fit_params):
        from sklearn.decomposition import FactorAnalysis
        from sklearn.impute import SimpleImputer

        X = X.to_numpy()
        imp = SimpleImputer()
        X = imp.fit_transform(X)
        n_components = self._n_components
        if min(X.shape) <= n_components:
            n_components = min(X.shape) - 1

        self.fa = FactorAnalysis(n_components=n_components)
        self.fa.fit(X)

        return self.fa.transform(X)

    def transform(self, X, y=None, **fit_params):
        from sklearn.impute import SimpleImputer

        X = X.to_numpy()
        imp = SimpleImputer()
        X = imp.fit_transform(X)

        return self.fa.transform(X)
