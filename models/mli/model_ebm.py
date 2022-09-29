"""Explainable Boosting Machines (EBM), implementation of GA2M with option for user-defined interaction between features. """
import datatable as dt
import numpy as np
import logging
from h2oaicore.models import CustomModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count, config

# Can be either an integer defining the number of allowed interactions or a
# list of lists of feature indices for which we allow interactions (see comment for example)
ALLOWED_INTERACTIONS = 1  # [[0, 1], [1, 2]]


class EBMModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False  # According to the `interpret` library: "Multiclass is still experimental. Subject to change per release." So, set to `True` at your own risk.
    # Current known issue(s): https://github.com/interpretml/interpret/issues/142
    _display_name = "EBM"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _description = (
        "Explainable Boosting Machines (EBM) are a faster implementation of GA2M. "
        "References:"
        "GA2M: "
        "Yin Lou, Rich Caruana, Johannes Gehrke, and Giles Hooker (2013). "
        "Accurate intelligible models with pairwise interactions. In The 19th ACM "
        "SIGKDD International Conference on Knowledge Discovery and Data Mining, "
        "KDD 2013, Chicago, IL, USA, August 11-14, 2013, pages 623â631, 2013. "
        "doi: 10.1145/2487575.2487579. URL https://doi.org/10.1145/2487575.2487579."
        "EBM: "
        "H. Nori, S. Jenkins, P. Koch, and R. Caruana (2019). InterpretML: A "
        "Unified Framework for Machine Learning Interpretability. "
        "URL https://arxiv.org/pdf/1909.09223.pdf"
    )
    _modules_needed_by_name = ["pillow==8.3.2", "interpret==0.1.20"]

    @staticmethod
    def do_acceptance_test():
        return False  # would fail for imbalanced binary problems when logloss gets constant response for holdout (EBM should be passing labels)

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        return False  # by default GA2M too slow, but if the only model selected this will still allow use

    def set_default_params(
            self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs
    ):
        # Fill up parameters we care about
        max_rounds = (
            min(kwargs.get("n_estimators", 100), 1000) if not config.hard_asserts else 1
        )
        self.params = dict(
            random_state=kwargs.get("random_state", 1234),
            max_rounds=max_rounds,
            interactions=ALLOWED_INTERACTIONS if self.num_classes <= 2 else 0,
            learning_rate=max(kwargs.get("learning_rate", 0.1), 0.0001),
            n_jobs=self.params_base.get("n_jobs", max(1, physical_cores_count)),
        )

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            estimators_list = [50, 100, 150, 200, 300, 400]
            learning_rate_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        elif accuracy >= 5:
            estimators_list = [30, 50, 100, 150, 200, 250]
            learning_rate_list = [0.02, 0.04, 0.06, 0.08, 0.09, 0.1]
        else:
            estimators_list = [30, 50, 100, 120, 150, 180, 200]
            learning_rate_list = [0.03, 0.04, 0.06, 0.1, 0.12, 0.15]

        # Modify certain parameters for tuning
        self.params["max_rounds"] = (
            int(np.random.choice(estimators_list)) if not config.hard_asserts else 1
        )
        self.params["learning_rate"] = float(np.random.choice(learning_rate_list))

    def get_importances(self, model, num_cols):
        ebm_global = model.explain_global(name="EBM")
        model.explain_global(name="EBM")
        names = ebm_global.data()["names"]
        scores = ebm_global.data()["scores"]
        importances = [0.0] * num_cols
        for jj in range(len(names)):
            if " x " not in names[jj]:
                idx = int(names[jj].replace("feature_", ""))
                if idx == num_cols:
                    idx -= 1
                importances[idx] += scores[idx]
            else:
                sub_features = names[jj].split(" x ")
                for feature in sub_features:
                    idx = int(feature.replace("feature_", ""))
                    if idx == num_cols:
                        idx -= 1
                    importances[idx] += scores[idx]
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
            10  # HACK - EBM can't handle our custom logger with unknown level 9 (DATA)
        )

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
            iterations=self.params["max_rounds"],
        )

    def basic_impute(self, X):
        # scikit extra trees internally converts to np.float32 during all operations,
        # so if float64 datatable, need to cast first, in case will be nan for float32
        from h2oaicore.systemutils import update_precision

        X = update_precision(
            X,
            data_type=np.float32,
            override_with_data_type=True,
            fixup_almost_numeric=True,
        )
        # Replace missing values with a value smaller than all observed values
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
