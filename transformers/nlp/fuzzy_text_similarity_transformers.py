"""Row-by-row similarity between two text columns based on FuzzyWuzzy"""
# https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# https://github.com/seatgeek/fuzzywuzzy 
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np

_global_modules_needed_by_name = ['nltk==3.4.1']
import nltk


class FuzzyBaseTransformer:
    _modules_needed_by_name = ['fuzzywuzzy==0.17.0']
    _method = NotImplemented
    _parallel_task = False

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from fuzzywuzzy import fuzz
        method = getattr(fuzz, self.__class__._method)
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:, 0].values
        text2_arr = X.iloc[:, 1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]
                text2 = str(text2).lower().split()
                ratio = method(text1, text2)
                output.append(ratio)
            except:
                output.append(-1)
        return np.array(output)


class FuzzyQRatioTransformer(FuzzyBaseTransformer, CustomTransformer):
    _method = "QRatio"


class FuzzyWRatioTransformer(FuzzyBaseTransformer, CustomTransformer):
    _method = "WRatio"


class FuzzyPartialRatioTransformer(FuzzyBaseTransformer, CustomTransformer):
    _method = "partial_ratio"


class FuzzyTokenSetRatioTransformer(FuzzyBaseTransformer, CustomTransformer):
    _method = "token_set_ratio"


class FuzzyTokenSortRatioTransformer(FuzzyBaseTransformer, CustomTransformer):
    _method = "token_sort_ratio"


class FuzzyPartialTokenSortRatioTransformer(FuzzyBaseTransformer, CustomTransformer):
    _method = "partial_token_sort_ratio"
