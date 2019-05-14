import importlib
# https://github.com/Mimino666/langdetect
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class TextLangDetectTransformer(CustomTransformer):
    
    _modules_needed_by_name = ['langdetect']
    
    language_codes = ['af', 'ar', 'bg', 'bn', 'ca', 'cs', 'cy', 'da', 'de',
                       'el', 'en', 'es', 'et', 'fa', 'fi', 'fr', 'gu', 'he',
                       'hi', 'hr', 'hu', 'id', 'it', 'ja', 'kn', 'ko', 'lt', 
                       'lv', 'mk', 'ml', 'mr', 'ne', 'nl', 'no', 'pa', 'pl',
                       'pt', 'ro', 'ru', 'sk', 'sl', 'so', 'sq', 'sv', 'sw', 
                       'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi', 'zh',
                       'zh-cn', 'zh-tw']
    
    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
    
    @staticmethod
    def detectLanguageAndEncode(s):
        #mod = importlib.import_module("langdetect")
        #detect_method = getattr(mod, "detect")
        #code = detect_method(s)
        from langdetect import detect
        code = detect(s)
        code_index = TextLangDetectTransformer.language_codes.index(code) if code in TextLangDetectTransformer.language_codes else -1
        return code_index
    
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(
                lambda x: self.detectLanguageAndEncode(x))
    