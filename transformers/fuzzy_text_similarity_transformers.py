# https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
# https://github.com/seatgeek/fuzzywuzzy 
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import nltk

class FuzzyQRatioTransformer(CustomTransformer):
    _modules_needed_by_name = ['fuzzywuzzy==0.17.0']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from fuzzywuzzy import fuzz
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:,0].values
        text2_arr = X.iloc[:,1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]
                text2 = str(text2).lower().split()
                ratio = fuzz.QRatio(text1, text2)
                output.append(ratio)
            except:
                output.append(-1)
        return np.array(output)

class FuzzyWRatioTransformer(CustomTransformer):
    _modules_needed_by_name = ['fuzzywuzzy==0.17.0']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from fuzzywuzzy import fuzz
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:,0].values
        text2_arr = X.iloc[:,1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]
                text2 = str(text2).lower().split()
                ratio = fuzz.WRatio(text1, text2)
                output.append(ratio)
            except:
                output.append(-1)
        return np.array(output)

class FuzzyPartialRatioTransformer(CustomTransformer):
    _modules_needed_by_name = ['fuzzywuzzy==0.17.0']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from fuzzywuzzy import fuzz
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:,0].values
        text2_arr = X.iloc[:,1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]
                text2 = str(text2).lower().split()
                ratio = fuzz.partial_ratio(text1, text2)
                output.append(ratio)
            except:
                output.append(-1)
        return np.array(output)

class FuzzyTokenSetRatioTransformer(CustomTransformer):
    _modules_needed_by_name = ['fuzzywuzzy==0.17.0']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from fuzzywuzzy import fuzz
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:,0].values
        text2_arr = X.iloc[:,1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]
                text2 = str(text2).lower().split()
                ratio = fuzz.token_set_ratio(text1, text2)
                output.append(ratio)
            except:
                output.append(-1)
        return np.array(output)

class FuzzyTokenSortRatioTransformer(CustomTransformer):
    _modules_needed_by_name = ['fuzzywuzzy==0.17.0']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from fuzzywuzzy import fuzz
        output = []     
        X = X.to_pandas()
        text1_arr = X.iloc[:,0].values
        text2_arr = X.iloc[:,1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]     
                text2 = str(text2).lower().split()
                ratio = fuzz.token_sort_ratio(text1, text2)
                output.append(ratio)
            except:
                output.append(-1)
        return np.array(output)

class FuzzyPartialTokenSortRatioTransformer(CustomTransformer):
    _modules_needed_by_name = ['fuzzywuzzy==0.17.0']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from fuzzywuzzy import fuzz
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:,0].values
        text2_arr = X.iloc[:,1].values      
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]
                text2 = str(text2).lower().split()
                ratio = fuzz.partial_token_sort_ratio(text1, text2)
                output.append(ratio)
            except:
                output.append(-1)
        return np.array(output)
