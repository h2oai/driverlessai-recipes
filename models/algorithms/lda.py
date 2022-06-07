"""Linear/Quadratic Discriminant Analysis (LDA/QDA) model from sklearn"""
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count, config, IgnoreEntirelyError


class DAModel(CustomModel):
    _regression = False
    _binary = True
    _multiclass = True
    _display_name = "Discriminant_Analysis"
    _description = "Discriminant Analysis (linear and quadratic) Model based on sklearn"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _parallel_task = False
    _supports_sample_weight = False

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0,
                num_classes=None, **kwargs):
        if config.hard_asserts:
            # for bigger data, too slow to test even with 1 iteration
            use = True
            use &= train_shape is not None and train_shape[0] * train_shape[1] < 1024 * 1024 or train_shape is None
            use &= valid_shape is not None and valid_shape[0] * valid_shape[1] < 1024 * 1024 or valid_shape is None
            use &= test_shape is not None and test_shape[0] * test_shape[1] < 1024 * 1024 or test_shape is None
            # too slow for walmart with only 421k x 15
            use &= train_shape is not None and train_shape[1] < 10 or train_shape is None
            return use
        else:
            return True

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = dict(solver='svd',
                           model_type='lda',
                           tol=1e-4,
                           reg_param=0.0,  # QDA only
                           )

        self.params_override()

    def mutate_params(self, accuracy=10, **kwargs):
        # Modify certain parameters for tuning
        if kwargs.get('train_shape') is not None and kwargs['train_shape'][1] < 10:
            solvers = ['svd', 'lsqr', 'eigen']
        else:
            solvers = ['svd']  # to avoid covariance matrix calculation for many features
        self.params["solver"] = str(np.random.choice(solvers))
        if accuracy >= 8:
            self.params['tol'] = 1e-4
        elif accuracy >= 5:
            self.params['tol'] = 1e-3
        else:
            self.params['tol'] = 1e-2
        model_types = ['lda', 'qda']
        self.params["model_type"] = str(np.random.choice(model_types))

        self.params['reg_param'] = np.random.choice([0.0, 0.1, 0.5])  # QDQ only

        self.params_override()

    def params_override(self):
        # override
        for k in self.params:
            if k in config.recipe_dict:
                self.params[k] = config.recipe_dict[k]

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)

        self.params_override()
        params = self.params.copy()
        if params.get('model_type', 'lda') == 'lda':
            model_class = LinearDiscriminantAnalysis
            params.pop('reg_param', None)
        else:
            model_class = QuadraticDiscriminantAnalysis
            params.pop('solver', None)
        params.pop('model_type', None)

        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = model_class(**params)
        else:
            model = model_class(**params)

        X = self.basic_impute(X)
        X = X.to_numpy()

        try:
            model.fit(X, y)
        except np.linalg.LinAlgError as e:
            # nothing can be done, just revert to constant predictions
            raise IgnoreEntirelyError(str(e))

        importances = np.array([1 for x in range(len(orig_cols))])
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=1)

    def basic_impute(self, X):
        # scikit extra trees internally converts to np.float32 during all operations,
        # so if float64 datatable, need to cast first, in case will be nan for float32
        from h2oaicore.systemutils import update_precision
        X = update_precision(X, data_type=np.float32, override_with_data_type=True, fixup_almost_numeric=True)
        # Replace missing values with a value smaller than all observed values
        if not hasattr(self, 'min'):
            self.min = dict()
        for col in X.names:
            XX = X[:, col]
            if col not in self.min:
                self.min[col] = XX.min1()
                if self.min[col] is None or np.isnan(self.min[col]) or np.isinf(self.min[col]):
                    self.min[col] = -1e10
                else:
                    self.min[col] -= 1
            XX.replace([None, np.inf, -np.inf], self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        return X

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        X = self.basic_impute(X)
        X = X.to_numpy()
        model, _, _, _ = self.get_model_properties()
        if self.num_classes == 1:
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)
        return preds
