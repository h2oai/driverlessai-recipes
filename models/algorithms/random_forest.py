"""Random Forest (RandomForest) model from sklearn"""
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count, config


class RandomForestModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "RandomForest"
    _description = "Random Forest Model based on sklearn"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _use_single_core_if_many = True

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0,
                num_classes=None, **kwargs):
        if config.hard_asserts:
            # for bigger data, too slow to test even with 1 iteration
            use = train_shape is not None and train_shape[0] * train_shape[1] < 1024 * 1024 or \
                  valid_shape is not None and valid_shape[0] * valid_shape[1] < 1024 * 1024
            # too slow for walmart with only 421k x 15
            use &= train_shape is not None and train_shape[1] < 10
            return use
        else:
            return True

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        n_estimators = min(kwargs.get("n_estimators", 100), 1000)
        if config.hard_asserts:
            # for testing avoid too many trees
            n_estimators = 10
        self.params = dict(random_state=kwargs.get("random_state", 1234),
                           n_estimators=n_estimators,
                           criterion="gini" if self.num_classes >= 2 else "mse",
                           n_jobs=self.params_base.get('n_jobs', max(1, physical_cores_count)),
                           max_depth=14,
                           min_samples_split=2,
                           min_samples_leaf=1,
                           oob_score=False,
                           )

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            estimators_list = [100, 200, 300, 500, 1000, 2000]
        elif accuracy >= 5:
            estimators_list = [50, 100, 200, 300, 400, 500]
        elif accuracy >= 3:
            estimators_list = [10, 50, 100]
        elif accuracy >= 2:
            estimators_list = [10, 50]
        else:
            estimators_list = [10]
        if config.hard_asserts:
            # for testing avoid too many trees
            estimators_list = [10]
        # Modify certain parameters for tuning
        self.params["n_estimators"] = int(np.random.choice(estimators_list))
        self.params["criterion"] = np.random.choice(["gini", "entropy"]) if self.num_classes >= 2 \
            else np.random.choice(["mse", "mae"])
        max_depth_list = [6, 10, 14, 20]
        self.params['max_depth'] = int(np.random.choice(max_depth_list))
        min_samples_split_list = [2, 4, 10]
        self.params['min_samples_split'] = int(np.random.choice(min_samples_split_list))
        min_samples_leaf_list = [1, 2, 5]
        self.params['min_samples_leaf'] = int(np.random.choice(min_samples_leaf_list))
        oob_score_list = [True, False]
        self.params['oob_score'] = bool(np.random.choice(oob_score_list))

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = RandomForestClassifier(**self.params)
        else:
            model = RandomForestRegressor(**self.params)

        X = self.basic_impute(X)
        X = X.to_numpy()

        model.fit(X, y)
        importances = np.array(model.feature_importances_)
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=self.params['n_estimators'])

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
