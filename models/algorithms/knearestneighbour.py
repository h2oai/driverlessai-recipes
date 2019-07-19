"""K-Nearest Neighbor implementation by sklearn. For small data (< 200k rows)."""
import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from h2oaicore.models import CustomModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge  # will be used to derive feature importances
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


class KNearestNeighbourModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True

    _display_name = "KNearestNeighbour"
    _description = "K Nearest Neighbour Model based on sklearn. Not adviced if the data is larger than 200K rows"

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        n_jobs = -1
        n_neighbors = min(kwargs['n_neighbors'], 1000) if 'n_neighbors' in kwargs else 10
        metric = kwargs['metric'] if "metric" in kwargs and kwargs['metric'] in ["minkowski",
                                                                                 "cityblock"] else "cityblock"
        self.params = {'n_neighbors': n_neighbors,
                       'metric': metric,
                       'weights': "uniform",
                       'n_jobs': n_jobs,  # -1 is not supported
                       }

    def mutate_params(self,
                      accuracy=10,
                      **kwargs):

        n_jobs = -1
        list_of_neibs = [10, 50, 100, 150, 200, 250, 300]

        if accuracy > 8:
            list_of_neibs = [100, 200, 300, 500, 1000, 2000]
        elif accuracy >= 5:
            list_of_neibs = [50, 100, 200, 300, 400, 500]

        index = np.random.randint(0, high=len(list_of_neibs))
        n_neighbors = list_of_neibs[index]

        metric = kwargs['metric'] if "metric" in kwargs and kwargs['metric'] in ["minkowski",
                                                                                 "cityblock"] else "cityblock"
        self.params = {'n_neighbors': n_neighbors,
                       'metric': metric,
                       'weights': "uniform",
                       'n_jobs': n_jobs,  # -1 is not supported
                       }
        # Default version is do no mutation
        # Otherwise, change self.params for this model

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        orig_cols = list(X.names)
        feature_model = Ridge(alpha=1., random_state=self.random_state)

        self.params['n_neighbors'] = min(self.params['n_neighbors'], X.shape[0])

        if self.num_classes >= 2:

            model = KNeighborsClassifier(n_neighbors=self.params['n_neighbors'], metric=self.params['metric'],
                                         weights=self.params['weights'], n_jobs=self.params['n_jobs'])
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
        else:
            model = KNeighborsRegressor(n_neighbors=self.params['n_neighbors'], metric=self.params['metric'],
                                        weights=self.params['weights'], n_jobs=self.params['n_jobs'])
        self.means = dict()
        self.scaler = StandardScaler()
        for col in X.names:
            XX = X[:, col]
            self.means[col] = XX.mean1()
            if self.means[col] is None:
                self.means[col] = 0
            XX.replace(None, self.means[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()
        X = self.scaler.fit_transform(X)
        feature_model.fit(X, y)
        model.fit(X, y)
        importances = np.array(abs(feature_model.coef_))

        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),  # abs(model.coef_[0])
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
                # preds = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            return preds
        else:
            raise NotImplementedError("No Shapley for K-nearest model")
