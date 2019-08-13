"""Extract sentiment from text using lexicon and rule-based sentiment analysis tool called VADER"""
# https://github.com/cjhutto/vaderSentiment
# https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f

import importlib
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd




class VaderSentimentTransformer(CustomTransformer):
    _modules_needed_by_name = ['vaderSentiment']

    
    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def sentimentAnalysis(s):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        return analyzer.polarity_scores(s)['compound']

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(
            lambda x: self.sentimentAnalysis(x))
