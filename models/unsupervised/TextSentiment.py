"""Extract sentiment from text using pretrained models from TextBlob"""
import importlib
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer
from h2oaicore.models import CustomUnsupervisedModel
import datatable as dt
import numpy as np


class TextSentimentTransformer(CustomUnsupervisedTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _modules_needed_by_name = ['nltk==3.4.3', 'textblob']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def sentimentAnalysis(s):
        from textblob import TextBlob
        analysis = TextBlob(s)
        return analysis.sentiment[0]

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(
            lambda x: self.sentimentAnalysis(x))


class TextSentimentModel(CustomUnsupervisedModel):
    _included_pretransformers = ['TextOriginalTransformer']
    _included_transformers = ['TextSentimentTransformer']
    _included_scorers = ['UnsupervisedScorer']
