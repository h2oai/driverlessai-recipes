"""Target-encode high cardinality categorical text by their first few characters in the string """
"""The str columns must be first marked as text in Data Sets page before recipe can take effect """

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from h2oaicore.transformers import CVTargetEncodeTransformer
from sklearn.preprocessing import LabelEncoder


class firstNChars:
    def fit_transform(self, X: dt.Frame, n):
        return self.transform(X, n)

    def transform(self, X: dt.Frame, n):
        assert X.ncols == 1
        return dt.Frame(X.to_pandas().apply(lambda x: x[0:n], axis=1))


class frst1ChrsCVTE(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _unsupervised = False  # uses target
    _uses_target = True  # uses target

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.binner = firstNChars()
        X = self.binner.fit_transform(X, 1)

        # Compute mean target (out of fold) per same string
        self.cvte = CVTargetEncodeTransformer(cat_cols=X.names)

        if self.labels is not None:
            # for classification, always turn y into numeric form, even if already integer
            y = dt.Frame(LabelEncoder().fit(self.labels).transform(y))

        X = self.cvte.fit_transform(X, y)
        return X

    def transform(self, X: dt.Frame):
        X = self.binner.transform(X, 1)
        X = self.cvte.transform(X)
        return X


class frst2ChrsCVTE(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _unsupervised = False  # uses target
    _uses_target = True  # uses target

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.binner = firstNChars()
        X = self.binner.fit_transform(X, 2)

        # Compute mean target (out of fold) per same string
        self.cvte = CVTargetEncodeTransformer(cat_cols=X.names)

        if self.labels is not None:
            # for classification, always turn y into numeric form, even if already integer
            y = dt.Frame(LabelEncoder().fit(self.labels).transform(y))

        X = self.cvte.fit_transform(X, y)
        return X

    def transform(self, X: dt.Frame):
        X = self.binner.transform(X, 2)
        X = self.cvte.transform(X)
        return X


class frst3ChrsCVTE(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _unsupervised = False  # uses target
    _uses_target = True  # uses target

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.binner = firstNChars()
        X = self.binner.fit_transform(X, 3)

        # Compute mean target (out of fold) per same string
        self.cvte = CVTargetEncodeTransformer(cat_cols=X.names)

        if self.labels is not None:
            # for classification, always turn y into numeric form, even if already integer
            y = dt.Frame(LabelEncoder().fit(self.labels).transform(y))

        X = self.cvte.fit_transform(X, y)
        return X

    def transform(self, X: dt.Frame):
        X = self.binner.transform(X, 3)
        X = self.cvte.transform(X)
        return X


class frst4ChrsCVTE(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _unsupervised = False  # uses target
    _uses_target = True  # uses target

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.binner = firstNChars()
        X = self.binner.fit_transform(X, 4)

        # Compute mean target (out of fold) per same string
        self.cvte = CVTargetEncodeTransformer(cat_cols=X.names)

        if self.labels is not None:
            # for classification, always turn y into numeric form, even if already integer
            y = dt.Frame(LabelEncoder().fit(self.labels).transform(y))

        X = self.cvte.fit_transform(X, y)
        return X

    def transform(self, X: dt.Frame):
        X = self.binner.transform(X, 4)
        X = self.cvte.transform(X)
        return X
