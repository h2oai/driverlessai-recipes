import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler


class ExtraTreesModel(CustomModel):

    _regression = True
    _binary = True
    _multiclass = True

    _display_name = "ExtraTrees"
    _description = "Extra Trees Model based on sklearn"

    # def make_instance(self):
    #     return self.__class__._class(seed=self.random_state)

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        n_jobs = -1
        n_estimators = min(kwargs.get('n_estimators', 10), 1000)
        criterion = kwargs['metric'] if "metric" in kwargs and kwargs["metric"] in ["gini", "entropy"] else "gini"
        self.params = {'n_estimators': n_estimators,
                       'criterion': criterion,
                       'n_jobs': n_jobs}

    def mutuate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        n_jobs = -1

        if accuracy > 8:
            estimators_list = [100, 200, 300, 500, 1000, 2000]
        elif accuracy >= 5:
            estimators_list = [50, 100, 200, 300, 400, 500]
        else:
            estimators_list = [10, 50, 100, 150, 200, 250, 300]

        criterion = kwargs['metric'] if "metric" in kwargs and kwargs['metric'] in ["mae", "mse", "gini", "entropy"] else "gini"

        index = np.random.randint(0, high=len(estimators_list))
        n_estimators = estimators_list[index]

        self.params = {"n_estimators": n_estimators,
                       "criterion": criterion,
                       "n_jobs": n_jobs}

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        orig_cols = list(X.names)
        if self.num_classes >= 2:
            param_criterion = self.params.get('criterion', 'gini')
            criterion = param_criterion if param_criterion in ['gini', 'entropy'] else 'gini'
            model = ExtraTreesClassifier(n_estimators=self.params['n_estimators'],
                                         criterion=criterion,
                                         n_jobs=self.params['n_jobs'],
                                         random_state=1)
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
        else:
            param_criterion = self.params.get('criterion', 'mse')
            criterion = param_criterion if param_criterion in ['mse', 'mae'] else 'mse'
            model = ExtraTreesRegressor(n_estimators=self.params['n_estimators'],
                                        criterion=criterion,
                                        n_jobs=self.params['n_jobs'],
                                        random_state=1)

        self.means = dict()
        self.scaler = StandardScaler()
        for col in X.names:
            XX = X[:, col]
            self.means[col] = XX.mean1()
            if np.isnan(self.means[col]):
                self.means[col] = 0
            XX.replace(None, self.means[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()
        X = self.scaler.fit_transform(X)

        model.fit(X, y)

        importances = np.array(model.feature_importances_)

        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=0)

    def predict(self, X, **kwargs):
        X = dt.Frame(X)

        for col in X.names:
            XX = X[:, col]
            XX.replace(None, self.means[col])
            X[:, col] = XX

        pred_contribs = kwargs.get('pred_contribs', None)
        output_margin = kwargs.get('output_margin', None)

        model, _, _, _ = self.get_model_properties()

        X = X.to_numpy()
        X = self.scaler.transform(X)
        if not pred_contribs:
            if self.num_classes == 1:
                preds = model.predict(X)
            else:
                preds = model.predict_proba(X)
            return preds
        else:
            raise NotImplementedError("No Shapley for ExtraTrees model")


