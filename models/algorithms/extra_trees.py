"""Extremely Randomized Trees (ExtraTrees) model from sklearn"""
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder


class ExtraTreesModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "ExtraTrees"
    _description = "Extra Trees Model based on sklearn"

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        n_jobs = -1
        n_estimators = min(kwargs.get("n_estimators", 10), 1000)
        self.params["n_estimators"] = n_estimators
        self.params["criterion"] = "gini" if self.num_classes >= 2 else "mse"
        self.params["n_jobs"] = n_jobs

    def mutate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        if accuracy > 8:
            estimators_list = [100, 200, 300, 500, 1000, 2000]
        elif accuracy >= 5:
            estimators_list = [50, 100, 200, 300, 400, 500]
        else:
            estimators_list = [10, 50, 100, 150, 200, 250, 300]
        self.params["n_estimators"] = int(np.random.choice(estimators_list))
        self.params["criterion"] = np.random.choice(["gini", "entropy"]) if self.num_classes >= 2 \
            else np.random.choice(["mse", "mae"])

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = ExtraTreesClassifier(random_state=1, **self.params)
        else:
            model = ExtraTreesRegressor(random_state=1, **self.params)

        self.min = dict()
        for col in X.names:
            XX = X[:, col]
            self.min[col] = XX.min1()
            if np.isnan(self.min[col]):
                self.min[col] = -1e10
            XX.replace(None, self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()

        model.fit(X, y)
        importances = np.array(model.feature_importances_)
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=len(model.estimators))

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        for col in X.names:
            XX = X[:, col]
            XX.replace(None, self.min[col])
            X[:, col] = XX
        model, _, _, _ = self.get_model_properties()
        X = X.to_numpy()
        if self.num_classes == 1:
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)
        return preds
