"""Text classification model using binary count of words"""
import random
import numpy as np
import scipy as sp
import datatable as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from h2oaicore.models import CustomModel
from h2oaicore.transformer_utils import CustomTransformer


class TextBinaryCountLogisticModel(CustomModel):
    """Text classification model using binary count of words"""
    _regression = False
    _binary = True
    _multiclass = True
    _can_handle_non_numeric = True
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _included_transformers = ["TextOriginalTransformer"]

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        self.params = dict(max_features=kwargs.get("max_features", None),
                           C=kwargs.get("C", 1.0),
                           max_iter=kwargs.get("max_iter", 100))

    def mutate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params["max_features"] = np.random.choice([20000, 50000, 100000, None])
        self.params["C"] = np.random.choice([0.1, 0.3, 1.0, 3.0, 10.0])
        self.params["max_iter"] = np.random.choice([100, 200])

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        lb = LabelEncoder()
        lb.fit(self.labels)
        y = lb.transform(y)
        model = LogisticRegression(C=self.params["C"],
                                   max_iter=self.params["max_iter"],
                                   fit_intercept=False,
                                   random_state=520)

        count_objs = []
        new_X = None
        for col in X.names:
            XX = X[:, col].to_pandas()
            XX = XX[col].astype(str).values.tolist()
            count_vec = CountVectorizer(max_features=self.params["max_features"],
                                        binary=True)
            XX = count_vec.fit_transform(XX)
            count_objs.append(count_vec)
            if new_X is None:
                new_X = XX
            else:
                new_X = sp.sparse.hstack([new_X, XX])

        model.fit(new_X, y)
        model = (model, count_objs)
        importances = [1] * len(orig_cols)
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances,
                                  iterations=0)

    def predict(self, X, **kwargs):
        (model, count_objs), _, _, _ = self.get_model_properties()
        X = dt.Frame(X)
        new_X = None
        for ind, col in enumerate(X.names):
            XX = X[:, col].to_pandas()
            XX = XX[col].astype(str).values.tolist()
            count_vec = count_objs[ind]
            XX = count_vec.transform(XX)
            if new_X is None:
                new_X = XX
            else:
                new_X = sp.sparse.hstack([new_X, XX])
        preds = model.predict_proba(new_X)
        return preds
