import importlib
# https://github.com/Mimino666/langdetect
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class TextSentimentTransformer(CustomTransformer):
    
    _modules_needed_by_name = ['textblob']
    
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
    