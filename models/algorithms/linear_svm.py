"""Linear Support Vector Machine (SVM) implementation by sklearn. For small data."""
import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from h2oaicore.models import CustomModel
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler


class linsvc(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1., penalty="l2", loss="squared_hinge", dual=True,
                 random_state=1
                 ):
        self.random_state = random_state
        self.C = C
        self.dual = dual
        self.loss = loss
        self.penalty = penalty
        self.model = LinearSVC(penalty=self.penalty, loss=self.loss, C=self.C, dual=self.dual,
                               random_state=random_state)

        self.classes_ = [0, 1]

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)

        return self

    def predict(self, X):  # this predicts classification

        preds = self.model.predict(X)
        return preds

    def predict_proba(self, X):
        X1 = X.dot(self.model.coef_[0])
        return np.column_stack((np.array(X1) - 1, np.array(X1)))

    def set_params(self, random_state=1, C=1., loss="squared_hinge", penalty="l2"):
        self.model.set_params(random_state=random_state, C=C, loss=loss, penalty=penalty)

    def get_params(self, deep=False):
        return {"random_state": self.random_state,
                "C": self.C,
                "loss": self.loss,
                "penalty": self.penalty,
                "dual": self.dual
                }

    def get_coeff(self):
        return self.model.coef_[0]


class LinearSVMModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False  # WIP

    _display_name = "LinearSVM"
    _description = "Linear Support Vector Machine with the Liblinear method + Calibration for probabilities"

    # LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss=’epsilon_insensitive’, fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
    # LinearSVC(penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’, fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):

        C = max(kwargs['C'], 0.00001) if 'C' in kwargs else 1.
        epsilon = max(kwargs['epsilon'], 0.00001) if 'epsilon' in kwargs else 0.1
        penalty = kwargs['penalty'] if "penalty" in kwargs and kwargs['penalty'] in ["l2", "l1"] else "l2"
        dual = True

        if self.num_classes >= 2:
            loss = kwargs['loss'] if "loss" in kwargs and kwargs['loss'] in ["squared_hinge",
                                                                             "hinge"] else "squared_hinge"
        else:
            base_loss = "squared_epsilon_insensitive"
            if self.params_base['score_f_name'] == "MAE" or self.params_base['score_f_name'] == "MAPE":
                base_loss = "epsilon_insensitive"
            loss = kwargs['loss'] if "loss" in kwargs and kwargs['loss'] in ["squared_epsilon_insensitive",
                                                                             "epsilon_insensitive"] else base_loss

        self.params = {'C': C,
                       'loss': loss,
                       'epsilon': epsilon,
                       'penalty': penalty,
                       'dual': dual,
                       }

    def mutate_params(self,
                      **kwargs):

        dual = True
        list_of_C = [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]
        list_of_loss = ["squared_epsilon_insensitive", "epsilon_insensitive"]
        if self.num_classes >= 2:
            list_of_loss = ["squared_hinge", "hinge"]
        list_of_epsilon = [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]
        list_of_penalty = ["l2", "l1"]

        C_index = np.random.randint(0, high=len(list_of_C))
        loss_index = np.random.randint(0, high=len(list_of_loss))
        epsilon_index = np.random.randint(0, high=len(list_of_epsilon))
        penalty_index = np.random.randint(0, high=len(list_of_penalty))

        C = list_of_C[C_index]
        loss = list_of_loss[loss_index]
        penalty = list_of_penalty[penalty_index]
        epsilon = list_of_epsilon[epsilon_index]

        if self.num_classes >= 2:
            if loss == "squared_hinge":
                dual = False
            elif loss == "hinge":
                if penalty == "l1":
                    penalty = "l2"

        self.params = {'C': C,
                       'loss': loss,
                       'epsilon': epsilon,
                       'penalty': penalty,
                       'dual': dual
                       }

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        orig_cols = list(X.names)

        if self.num_classes >= 2:
            mod = linsvc(random_state=self.random_state, C=self.params["C"], penalty=self.params["penalty"],
                         loss=self.params["loss"], dual=self.params["dual"])
            kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            model = CalibratedClassifierCV(base_estimator=mod, method='isotonic', cv=kf)
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
        else:
            model = LinearSVR(epsilon=self.params["epsilon"], C=self.params["C"], loss=self.params["loss"],
                              dual=self.params["dual"], random_state=self.random_state)
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
        model.fit(X, y, sample_weight=sample_weight)
        if self.num_classes >= 2:
            importances = np.array([0.0 for k in range(len(orig_cols))])
            for classifier in model.calibrated_classifiers_:
                importances += np.array(abs(classifier.base_estimator.get_coeff()))
        else:
            importances = np.array(abs(model.coef_[0]))

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
            raise NotImplementedError("No Shapley for SVM")
