"""Character level TFIDF and Count followed by Truncated SVD on text columns"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class TextCharTFIDFTransformer(CustomTransformer):
    _unsupervised = True

    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(self, max_ngram, n_svd_comp, **kwargs):
        super().__init__(**kwargs)
        self.max_ngram = max_ngram
        self.n_svd_comp = n_svd_comp

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"max_ngram": [3, 2, 1],
                "n_svd_comp": [50, 20, 100]}

    @property
    def display_name(self):
        return f"CharTFIDF_{self.max_ngram}maxgram_SVD_{self.n_svd_comp}comp"

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        X = X.to_pandas().astype(str).iloc[:, 0].fillna("NA")
        # TFIDF Vectorizer
        self.tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(1, self.max_ngram))
        X = self.tfidf_vec.fit_transform(X)
        # Truncated SVD
        if len(self.tfidf_vec.vocabulary_) <= self.n_svd_comp:
            self.n_svd_comp = len(self.tfidf_vec.vocabulary_) - 1
        self.truncated_svd = TruncatedSVD(n_components=self.n_svd_comp, random_state=2019)
        X = self.truncated_svd.fit_transform(X)
        return X

    def transform(self, X: dt.Frame):
        X = X.to_pandas().astype(str).iloc[:, 0].fillna("NA")
        X = self.tfidf_vec.transform(X)
        X = self.truncated_svd.transform(X)
        return X


class TextCharCountTransformer(CustomTransformer):
    _unsupervised = True

    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(self, max_ngram, n_svd_comp, **kwargs):
        super().__init__(**kwargs)
        self.max_ngram = max_ngram
        self.n_svd_comp = n_svd_comp

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"max_ngram": [3, 2, 1],
                "n_svd_comp": [50, 20, 100]}

    @property
    def display_name(self):
        return f"CharCount_max{self.max_ngram}gram_SVD_{self.n_svd_comp}comp"

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        X = X.to_pandas().astype(str).iloc[:, 0].fillna("NA")
        # Count Vectorizer
        self.cnt_vec = CountVectorizer(analyzer="char", ngram_range=(1, self.max_ngram))
        X = self.cnt_vec.fit_transform(X)
        # Truncated SVD
        if len(self.cnt_vec.vocabulary_) <= self.n_svd_comp:
            self.n_svd_comp = len(self.cnt_vec.vocabulary_) - 1
        self.truncated_svd = TruncatedSVD(n_components=self.n_svd_comp, random_state=2019)
        X = self.truncated_svd.fit_transform(X)
        return X

    def transform(self, X: dt.Frame):
        X = X.to_pandas().astype(str).iloc[:, 0].fillna("NA")
        X = self.cnt_vec.transform(X)
        X = self.truncated_svd.transform(X)
        return X
