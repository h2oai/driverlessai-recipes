"""Text classification / regression model using TFIDF"""
import random
import numpy as np
import scipy as sp
import datatable as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from h2oaicore.models import CustomModel
from h2oaicore.transformer_utils import CustomTransformer


class TextTFIDFModel(CustomModel):
    """Text classification / regression model using TFIDF"""
    _regression = True
    _binary = True
    _multiclass = True
    _can_handle_non_numeric = True
    _can_handle_text = True
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _included_transformers = ["TextOriginalTransformer"]

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        self.params = dict(max_features=kwargs.get("max_features", None),
                           ngram_range=kwargs.get("ngram_range", (1, 1)))

    def mutate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params["max_features"] = np.random.choice([50000, 100000, None])
        self.params["ngram_range"] = random.choice([(1, 1), (1, 2), (1, 3)])

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        text_names = X[:, [str]].names
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            # somehow seeing: solver lbfgs supports only \\'l2\\' or \\'none\\' penalties, got l1 penalty.
            model = LogisticRegression(random_state=2019, solver='lbfgs', penalty='l2')
        else:
            model = LinearRegression()

        self.tfidf_objs = {}
        new_X = None
        for col in text_names:
            XX = X[:, col].to_pandas()
            XX = XX[col].astype(str).fillna("NA").values.tolist()
            tfidf_vec = TfidfVectorizer(**self.params)
            try:
                XX = tfidf_vec.fit_transform(XX)
            except ValueError as e:
                if 'vocab' in str(e):
                    # skip non-text-like column
                    continue
                else:
                    raise

            self.tfidf_objs[col] = tfidf_vec
            if new_X is None:
                new_X = XX
            else:
                new_X = sp.sparse.hstack([new_X, XX])

        model.fit(new_X, y)
        model = (model, self.tfidf_objs)
        self.tfidf_objs = {}
        importances = [1] * len(orig_cols)
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances,
                                  iterations=0)

    def predict(self, X, **kwargs):
        (model, tfidf_objs), _, _, _ = self.get_model_properties()
        X = dt.Frame(X)
        new_X = None
        text_names = X[:, [str]].names
        for col in text_names:
            if col  not in tfidf_objs:
                continue
            XX = X[:, col].to_pandas()
            XX = XX[col].astype(str).fillna("NA").values.tolist()
            tfidf_vec = tfidf_objs[col]
            XX = tfidf_vec.transform(XX)
            if new_X is None:
                new_X = XX
            else:
                new_X = sp.sparse.hstack([new_X, XX])
        if self.num_classes == 1:
            preds = model.predict(new_X)
        else:
            preds = model.predict_proba(new_X)
        return preds
