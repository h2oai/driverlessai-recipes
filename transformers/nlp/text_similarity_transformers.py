"""Row-by-row similarity between two text columns based on common N-grams, Jaccard similarity, Dice similarity and edit distance."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np

_global_modules_needed_by_name = ['nltk==3.9.1']
import nltk


class CountCommonNGramsTransformer(CustomTransformer):
    _unsupervised = True

    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(self, ngrams, **kwargs):
        super().__init__(**kwargs)
        self.ngrams = ngrams

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"ngrams": [1, 2, 3]}

    @property
    def display_name(self):
        return "CountCommon%dGrams" % self.ngrams

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:, 0].values
        text2_arr = X.iloc[:, 1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = set(nltk.ngrams(str(text1).lower().split(), self.ngrams))
                text2 = text2_arr[ind]
                text2 = set(nltk.ngrams(str(text2).lower().split(), self.ngrams))
                output.append(len(text1.intersection(text2)))
            except:
                output.append(-1)
        return np.array(output)


class JaccardSimilarityTransformer(CustomTransformer):
    _unsupervised = True

    """Jaccard similarity measure on n-grams"""

    def __init__(self, ngrams, **kwargs):
        super().__init__(**kwargs)
        self.ngrams = ngrams

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"ngrams": [1, 2, 3]}

    @property
    def display_name(self):
        return "JaccardSimilarity_%dGrams" % self.ngrams

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:, 0].values
        text2_arr = X.iloc[:, 1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = set(nltk.ngrams(str(text1).lower().split(), self.ngrams))
                text2 = text2_arr[ind]
                text2 = set(nltk.ngrams(str(text2).lower().split(), self.ngrams))
                output.append(len(text1.intersection(text2)) / len(text1.union(text2)))
            except:
                output.append(-1)
        return np.array(output)


class DiceSimilarityTransformer(CustomTransformer):
    _unsupervised = True

    """Dice similarity measure on n-grams"""

    def __init__(self, ngrams, **kwargs):
        super().__init__(**kwargs)
        self.ngrams = ngrams

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"ngrams": [1, 2, 3]}

    @property
    def display_name(self):
        return "DiceSimilarity_%dGrams" % self.ngrams

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:, 0].values
        text2_arr = X.iloc[:, 1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = set(nltk.ngrams(str(text1).lower().split(), self.ngrams))
                text2 = text2_arr[ind]
                text2 = set(nltk.ngrams(str(text2).lower().split(), self.ngrams))
                output.append((2 * len(text1.intersection(text2))) / (len(text1) + len(text2)))
            except:
                output.append(-1)
        return np.array(output)


class EditDistanceTransformer(CustomTransformer):
    _unsupervised = True

    _modules_needed_by_name = ['editdistance==0.8.1']

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        import editdistance
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:, 0].values
        text2_arr = X.iloc[:, 1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = str(text1).lower().split()
                text2 = text2_arr[ind]
                text2 = str(text2).lower().split()
                edit_distance = editdistance.eval(text1, text2)
                output.append(edit_distance)
            except:
                output.append(-1)
        return np.array(output)
