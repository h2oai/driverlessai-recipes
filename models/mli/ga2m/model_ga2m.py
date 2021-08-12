"""Explainable Boosting Machines (EBM), implementation of GA2M"""
import datatable as dt
import numpy as np
import logging
from h2oaicore.models import CustomModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count, config


class GA2MModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False  # According to the `interpret` library: "Multiclass is still experimental. Subject to change per release." So, set to `True` at your own risk.
    # Current known issue(s): https://github.com/interpretml/interpret/issues/142
    _display_name = "GA2M"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _description = (
        "GA2M Model. see: Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M. and Elhadad, N., 2015, August."
        "Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission."
        "In Proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1721-1730)."
    )
    _modules_needed_by_name = ['pillow==8.3.1', "interpret==0.1.20"]

    @staticmethod
    def do_acceptance_test():
        return (
            False
        )  # would fail for imbalanced binary problems when logloss gets constant response for holdout (EBM should be passing labels)

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        return False  # by default GA2M too slow, but if the only model selected this will still allow use

    def set_default_params(
            self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs
    ):
        # Fill up parameters we care about
        n_estimators = min(kwargs.get("n_estimators", 100), 1000) if not config.hard_asserts else 1
        max_tree_splits = min(kwargs.get("max_tree_splits", 10), 200) if not config.hard_asserts else 3
        self.params = dict(
            random_state=kwargs.get("random_state", 1234),
            n_estimators=n_estimators,
            interactions=1 if self.num_classes <= 2 else 0,
            max_tree_splits=max_tree_splits,
            learning_rate=max(kwargs.get("learning_rate", 0.1), 0.0001),
            n_jobs=self.params_base.get("n_jobs", max(1, physical_cores_count)),
        )

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            estimators_list = [50, 100, 150, 200, 300, 400]
            max_tree_splits_list = [10, 20, 30, 50, 80, 100]
            learning_rate_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        elif accuracy >= 5:
            estimators_list = [30, 50, 100, 150, 200, 250]
            max_tree_splits_list = [10, 20, 30, 30, 60, 80]
            learning_rate_list = [0.02, 0.04, 0.06, 0.08, 0.09, 0.1]
        else:
            estimators_list = [30, 50, 100, 120, 150, 180, 200]
            max_tree_splits_list = [5, 10, 20, 25, 30, 50]
            learning_rate_list = [0.03, 0.04, 0.06, 0.1, 0.12, 0.15]

        # Modify certain parameters for tuning
        self.params["n_estimators"] = int(np.random.choice(estimators_list)) if not config.hard_asserts else 1
        self.params["max_tree_splits"] = int(np.random.choice(max_tree_splits_list)) if not config.hard_asserts else 3
        self.params["learning_rate"] = float(np.random.choice(learning_rate_list))

    def get_importances(self, model, num_cols):
        ebm_global = model.explain_global(name="EBM")
        model.explain_global(name="EBM")
        names = ebm_global.data()["names"]
        scores = ebm_global.data()["scores"]
        importances = [0.0] * num_cols
        for jj in range(len(names)):
            if " x " not in names[jj]:
                importances[int(names[jj].replace("feature_", ""))] += scores[jj]
            else:
                sub_features = names[jj].split(" x ")
                for feature in sub_features:
                    importances[int(feature.replace("feature_", ""))] += scores[jj]
        return importances

    def fit(
            self,
            X,
            y,
            sample_weight=None,
            eval_set=None,
            sample_weight_eval_set=None,
            **kwargs
    ):
        from interpret.glassbox import (
            ExplainableBoostingClassifier,
            ExplainableBoostingRegressor,
        )

        logging.root.level = (
            10
        )  # HACK - EBM can't handle our custom logger with unknown level 9 (DATA)

        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = ExplainableBoostingClassifier(**self.params)
        else:
            model = ExplainableBoostingRegressor(**self.params)

        X = self.basic_impute(X)
        X = X.to_numpy()

        model.fit(X, y)
        importances = self.get_importances(model, X.shape[1])
        self.set_model_properties(
            model=model,
            features=orig_cols,
            importances=importances,
            iterations=self.params["n_estimators"],
        )

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
