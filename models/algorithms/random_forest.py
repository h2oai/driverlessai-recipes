"""Random Forest (RandomForest) model from sklearn"""
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count


class RandomForestModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "RandomForest"
    _description = "Random Forest Model based on sklearn"

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = dict(random_state=kwargs.get("random_state", 1234),
                           n_estimators=min(kwargs.get("n_estimators", 100), 1000),
                           criterion="gini" if self.num_classes >= 2 else "mse",
                           n_jobs=self.params_base.get('n_jobs', max(1, physical_cores_count)))

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            estimators_list = [100, 200, 300, 500, 1000, 2000]
        elif accuracy >= 5:
            estimators_list = [50, 100, 200, 300, 400, 500]
        else:
            estimators_list = [10, 50, 100, 150, 200, 250, 300]
        # Modify certain parameters for tuning
        self.params["n_estimators"] = int(np.random.choice(estimators_list))
        self.params["criterion"] = np.random.choice(["gini", "entropy"]) if self.num_classes >= 2 \
            else np.random.choice(["mse", "mae"])

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = RandomForestClassifier(**self.params)
        else:
            model = RandomForestRegressor(**self.params)

        # Replace missing values with a value smaller than all observed values
        self.min = dict()
        for col in X.names:
            XX = X[:, col]
            self.min[col] = XX.min1()
            if self.min[col] is None or np.isnan(self.min[col]):
                self.min[col] = -1e10
            else:
                self.min[col] -= 1
            XX.replace(None, self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()

        model.fit(X, y)
        importances = np.array(model.feature_importances_)
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=self.params['n_estimators'])

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
