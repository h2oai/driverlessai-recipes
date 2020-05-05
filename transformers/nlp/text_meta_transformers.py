"""Extract common meta features from text"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import string


class WordBaseTransformer:
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)


class CountWordsTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: len(x.split()))


class CountUniqueWordsTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: len(set(x.split())))


class CountUpperWordsTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: len([w for w in x.split() if w.isupper()]))


class CountNumericWordsTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: len([w for w in x.split() if w.isnumeric()]))


class CountUpperCharsTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: len([c for c in x if c.isupper()]))


class CountNumericCharsTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: len([c for c in x if c.isnumeric()]))


class CountPunctCharsTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: len([c for c in x if c in string.punctuation]))


class MeanWordLengthTransformer(WordBaseTransformer, CustomTransformer):
    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
