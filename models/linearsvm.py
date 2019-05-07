from h2oaicore.models import CustomModel
import datatable as dt
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

class LinearSVMModel(CustomModel):
    _regression = True
    _binary = True

    _boosters = ['linearsvm']
    _display_name = "LinearSVM"
    _description = "Linear Support Vector Machine"

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        orig_cols = list(X.names)
        from sklearn.svm import SVC, SVR
        if self.num_classes >= 2:
            model = SVC(kernel='linear', probability=True, random_state=self.random_state)
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
        else:
            model = SVR(kernel='linear')
        self.means = dict()
        for col in X.names:
            XX = X[:, col]
            self.means[col] = XX.mean1()
            if np.isnan(self.means[col]):
                self.means[col] = 0
            XX.replace(None, self.means[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()
        model.fit(X, y, sample_weight=sample_weight)

        # need to move to wrapper
        self.feature_names_fitted = orig_cols
        self.transformed_features = self.feature_names_fitted
        self.best_ntree_limit = 0
        # must always set best_iterations
        self.best_iterations = self.best_ntree_limit + 1

        self.set_feature_importances(abs(model.coef_[0]))
        self.model_bytes = pickle.dumps(model, protocol=4)
        self.model = None
        return self

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        for col in X.names:
            XX = X[:, col]
            XX.replace(None, self.means[col])
            X[:, col] = XX

        pred_contribs = kwargs.get('pred_contribs', None)
        output_margin = kwargs.get('output_margin', None)

        self.get_model()
        if not pred_contribs:
            if self.num_classes == 1:
                preds = self.model.predict(X.to_numpy())
            else:
                prob_pos = self.model.decision_function(X.to_numpy())
                preds = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            return preds
        else:
            raise NotImplementedError("No Shapley for SVM")
