"""Quantile Random Forest Regression model from skgarden"""
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from skgarden import RandomForestQuantileRegressor
from h2oaicore.systemutils import physical_cores_count


class RandomForestQuantileModel(CustomModel):
    _regression = True
    _binary = False
    _multiclass = False

    _alpha = 0.8  # PLEASE CONFIGURE

    _display_name = "QuantileRandomForest alpha=%g" % _alpha
    _description = "Quantile Random Forest Regression"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _modules_needed_by_name=['scikit-garden==0.1.3',] # extra packages required

    def set_default_params(
        self, 
        accuracy=None, 
        time_tolerance=None, 
        interpretability=None, 
        **kwargs
    ):
        # fill up parameters we care about
        self.params = dict(
            random_state=kwargs.get("random_state", 1234),
            n_estimators=min(kwargs.get("n_estimators", 100), 2000),
            criterion="mse",
            max_depth=10,
            min_samples_leaf=10,
            n_jobs=self.params_base.get("n_jobs", max(1, physical_cores_count)),
        )

    def mutate_params(
        self, 
        accuracy=10,
        **kwargs
    ):
        if accuracy > 8:
            estimators_list = [300, 500, 1000, 2000,]
            depth_list = [10, 20, 30, 50, 100,]
            samples_leaf_list = [10, 20, 30,]
        elif accuracy >= 5:
            estimators_list = [50, 100, 200, 300,]
            depth_list = [5, 10, 15, 25, 50,]
            samples_leaf_list = [20, 40, 60,]
        else:
            estimators_list = [10, 20, 40, 60,]
            depth_list = [1, 2, 3, 5, 10,]
            samples_leaf_list = [30, 60, 90,]

        criterion_list = ["mse", "mae",]

        # modify certain parameters for tuning
        self.params["n_estimators"] = int(np.random.choice(estimators_list))
        self.params["criterion"] = np.random.choice(criterion_list)
        self.params["max_depth"] = int(np.random.choice(depth_list))
        self.params["min_samples_leaf"] = int(np.random.choice(samples_leaf_list))

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        sample_weight_eval_set=None,
        **kwargs
    ):
        X = dt.Frame(X)
        orig_cols = list(X.names)

        model = RandomForestQuantileRegressor(**self.params)
        X = self.basic_impute(X)
        X = X.to_numpy()

        model.fit(X, y)
        importances = np.array(model.feature_importances_)
        self.set_model_properties(
            model=model,
            features=orig_cols,
            importances=importances.tolist(),
            iterations=self.params["n_estimators"],
        )

    def basic_impute(
        self, 
        X
    ):
        # scikit extra trees internally converts to np.float32 during all operations,
        # so if float64 datatable, need to cast first, in case will be nan for float32
        from h2oaicore.systemutils import update_precision

        X = update_precision(X, data_type=np.float32)
        # replace missing values with a value smaller than all observed values
        if not hasattr(self, "min"):
            self.min = dict()
        for col in X.names:
            XX = X[:, col]
            if col not in self.min:
                self.min[col] = XX.min1()
                if (
                    self.min[col] is None
                    or np.isnan(self.min[col])
                    or np.isinf(self.min[col])
                ):
                    self.min[col] = -1e10
                else:
                    self.min[col] -= 1
            XX.replace([None, np.inf, -np.inf], self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        return X

    def predict(
        self,
        X, 
        **kwargs
    ):
        X = dt.Frame(X)
        X = self.basic_impute(X)
        X = X.to_numpy()
        model, _, _, _ = self.get_model_properties()
        preds = model.predict(X, quantile=RandomForestQuantileModel._alpha)
        return preds
