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


# Text column should be passed through this transformer for the TextTFIDF model
class TextIdentityTransformer(CustomTransformer):
    """Identity transformer for text"""
    _numeric_output = False

    @property
    def display_name(self):
        return "Str"

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str)


class TextTFIDFModel(CustomModel):
    """Text classification / regression model using TFIDF"""
    _regression = True
    _binary = True
    _multiclass = True
    _can_handle_non_numeric = True

    _included_transformers = ["TextIdentityTransformer"]  # Takes input only from above transformer

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        self.params = dict(max_features=kwargs.get("max_features", None),
                           ngram_range=kwargs.get("ngram_range", (1, 1)))

    def mutate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params["max_features"] = np.random.choice([50000, 100000, None])
        self.params["ngram_range"] = random.choice([(1, 1), (1, 2), (1, 3)])

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = LogisticRegression(random_state=2019)
        else:
            model = LinearRegression()

        self.tfidf_objs = []
        new_X = None
        for col in X.names:
            XX = X[:, col].to_pandas()
            XX = XX[col].astype(str).fillna("NA").values.tolist()
            tfidf_vec = TfidfVectorizer(**self.params)
            XX = tfidf_vec.fit_transform(XX)
            self.tfidf_objs.append(tfidf_vec)
            if new_X is None:
                new_X = XX
            else:
                new_X = sp.sparse.hstack([new_X, XX])

        model.fit(new_X, y)
        importances = [1] * len(orig_cols)
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances,
                                  iterations=0)

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        new_X = None
        for ind, col in enumerate(X.names):
            XX = X[:, col].to_pandas()
            XX = XX[col].astype(str).fillna("NA").values.tolist()
            tfidf_vec = self.tfidf_objs[ind]
            XX = tfidf_vec.transform(XX)
            if new_X is None:
                new_X = XX
            else:
                new_X = sp.sparse.hstack([new_X, XX])
        model, _, _, _ = self.get_model_properties()
        if self.num_classes == 1:
            preds = model.predict(new_X)
        else:
            preds = model.predict_proba(new_X)
        return preds
