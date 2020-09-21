"""Explainable Text transformer that uses binary counts of words using sklearn's CountVectorizer"""

import datatable as dt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from h2oaicore.transformer_utils import CustomTransformer


# _global_modules_needed_by_name = []  # Optional global package requirements, for multiple custom recipes in a file


class TextBinaryCountTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _regression = True  # y has shape (N,) and is of numeric type, no missing values
    _binary = True  # y has shape (N,) and can be numeric or string, cardinality 2, no missing values
    _multiclass = True  # y has shape (N,) and can be numeric or string, cardinality 3+, no missing values
    _numeric_output = True
    _is_reproducible = True  # This transformer is deterministic
    _display_name = 'TextBinaryCountTransformer'
    _allow_transform_to_modify_output_feature_names = True  # Enable setting feature names/self._output_feature_names

    def __init__(self, max_ngram, max_tokens, max_df, do_binary, do_lowercase, remove_stopwords, **kwargs):
        super().__init__(**kwargs)
        self.max_ngram = max_ngram  # Max ngram size to be extracted
        self.max_tokens = max_tokens  # Max number of tokens/ngrams used in text (using highest frequency)
        self.max_df = max_df  # Document frequency threshold that tokens would be ignored
        self.do_binary = do_binary  # Binary counts of tokens (either 0 or 1)
        self.do_lowercase = do_lowercase  # Convert text to lowercase before tokenization
        self.remove_stopwords = remove_stopwords  # Turn on using 'english' as needed

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="text",
                    min_cols=1,
                    max_cols=1,
                    relative_importance=1  # relative importance to other transformer during feature evolution
                    )

    @staticmethod
    def get_parameter_choices():
        """
        Parameters for the initializer.
        Driverless AI will automatically sample (uniformly) from the values for each key. You will need to
        add repeated values to enforce non-uniformity of returned values, if desired.
        """
        return {"max_ngram": [1, 2, 3],
                "max_tokens": [20000, 50000, 100000, None],
                "max_df": [0.8, 0.9, 1.0],
                "do_binary": [True],
                "do_lowercase": [True],
                "remove_stopwords": [None]}

    @property
    def display_name(self):
        return f"TextBinaryCount_{self.max_ngram}maxngram_{self.max_tokens}maxtokens"

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        curr_col = X.names[0]  # Get name of current feature/col
        X = X.to_pandas().astype(str).iloc[:, 0].fillna("NA")
        # Count Vectorizer
        self.count_vec = CountVectorizer(ngram_range=(1, self.max_ngram),
                                         max_features=self.max_tokens,
                                         max_df=self.max_df,
                                         binary=self.do_binary,
                                         lowercase=self.do_lowercase,
                                         stop_words=self.remove_stopwords
                                         )
        X = self.count_vec.fit_transform(X).toarray()
        self._output_feature_names = ['BinaryCount:' + curr_col + '.' + token.replace(' ', '_') for token in
                                      self.count_vec.get_feature_names()]
        self._feature_desc = ["Binary count of '" + token + "' found in " + curr_col for token in
                              self.count_vec.get_feature_names()]
        return X

    def transform(self, X: dt.Frame):
        X = X.to_pandas().astype(str).iloc[:, 0].fillna("NA")
        X = self.count_vec.transform(X).toarray()
        return X
