"""Nu-SVM implementation by sklearn. For small data."""
import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from h2oaicore.models import CustomModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVC, NuSVR


class NuSVMModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False #WIP

    _boosters = ['nusvm']
    _display_name = "NuSVM"
    _description = "Nu-SVM model based on sklearn. Not advised for large data."

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        nu = min(kwargs['nu'], 1) if 'nu' in kwargs else 0.5
        kernel = kwargs['kernel'] if "kernel" in kwargs and kwargs['kernel'] in ['linear',
                                                                                 'rbf',
                                                                                 'poly',
                                                                                 'sigmoid'] else 'rbf'
        degree = kwargs['degree'] if 'degree' in kwargs and kwargs['degree'] in [3, 4, 5, 6] else 3
        self.params = {'nu': nu,
                       'kernel': kernel,
                       'degree': degree,
                       'probability': True
                       }

    def mutate_params(self,
                      accuracy, time_tolerance, interpretability,
                      **kwargs):

        n_jobs = -1
        list_of_nus = [0.45, 0.5, 0.55]
        list_of_kernels = ['linear', 'rbf']
        list_of_degrees = [3, 4]

        if accuracy > 8:
            list_of_neibs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            list_of_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
            list_of_degrees = [3, 4, 5, 6]
        elif accuracy >= 5:
            list_of_neibs = [0.2, 0.4, 0.5, 0.6, 0.8]

        nu_index = np.random.randint(0, high=len(list_of_nus))
        kernel_index = np.random.randint(0, high=len(list_of_kernels))
        degree_index = np.random.randint(0, high=len(list_of_degrees))
        
        nu = list_of_nus[nu_index]
        kernel = list_of_kernels[kernel_index]
        degree = list_of_degrees[degree_index]

        self.params = {'nu': nu,
                       'kernel': kernel,
                       'degree': degree,
                       'probability': True
                       }

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        orig_cols = list(X.names)

        if self.num_classes >= 2:
            feature_model = NuSVC(kernel='linear')
            model = NuSVC(nu=self.params['nu'], kernel=self.params['kernel'],
                      degree=self.params['degree'], probability=self.params['probability'])

            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
        else:
            feature_model = NuSVR(kernel='linear')
            model = NuSVR(nu=self.params['nu'], kernel=self.params['kernel'],
                      degree=self.params['degree'])

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
        feature_model.fit(X, y)
        model.fit(X, y)
        importances = np.array(abs(feature_model.coef_)).ravel()

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
        if not pred_contribs:
            if self.num_classes >= 2:
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X)
            return preds
        else:
            raise NotImplementedError("No Shapley for Nu-SVM model")
