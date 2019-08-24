"""Sklearn Logistic Regression."""
import datatable as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import  StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


from h2oaicore.models import CustomModel
from h2oaicore.systemutils import config, physical_cores_count
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
from h2oaicore.transformers import CatOriginalTransformer


class LogisticRegressionModel(CustomModel):
    _regression = False
    _binary = True
    _multiclass = False
    _can_handle_non_numeric = True
    _display_name = "LR"
    _description = "Logistic Regression"

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = dict(random_state=kwargs.get("random_state", 1234),
                           solver='lbfgs',
                           penalty='l2',
                           C=0.1,
                           n_jobs=self.params_base.get('n_jobs', max(1, physical_cores_count)))

    def mutate_params(self, accuracy=10, **kwargs):
        # Modify certain parameters for tuning
        C_list = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 10.0]
        self.params["C"] = float(np.random.choice(C_list))
        tol_list = [1e-4, 1e-3, 1e-5]
        self.params["tol"] = float(np.random.choice(tol_list))
        solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        self.params["solver"] = str(np.random.choice(solver_list))
        max_iter_list = [100, 200, 1000]
        self.params["max_iter"] = float(np.random.choice(max_iter_list))
        if self.params["solver"] == 'lbfgs':
            penalty_list = ['l2', 'none']
        else:
            penalty_list = ['l1', 'l2', 'elasticnet', 'none']
        self.params["penalty"] = str(np.random.choice(penalty_list))
        if self.params["penalty"] == 'elasticnet':
            l1_ratio_list = [0, 0.5, 1.0]
            self.params["l1_ratio"] = float(np.random.choice(l1_ratio_list))

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        if self.num_classes == 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)

        # Replace missing values with a value smaller than all observed values
        self.min = dict()
        for col in X.names:
            XX = X[:, col]
            self.min[col] = XX.min1()
            if self.min[col] is None or np.isnan(self.min[col]):
                self.min[col] = -1e10
            else:
                self.min[col] -= 1
            XX.replace(None, self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_pandas()
        X_names = list(X.columns)

        #cat_features = [x for x in X_names if CatOriginalTransformer.is_me_transformed(x)]
        #noncat_features = [x for x in X_names if x not in cat_features]
        numerical_features = X.dtypes == 'float'
        categorical_features = ~numerical_features
        preprocess = make_column_transformer(
            (numerical_features, make_pipeline(SimpleImputer(), StandardScaler())),
            (categorical_features, OneHotEncoder(sparse=True))
         )
        model = make_pipeline(
            preprocess,
            LogisticRegression(**self.params))

        grid_search = False
        if grid_search:
            from sklearn.model_selection import GridSearchCV
            param_grid = {
                'columntransformer__pipeline__simpleimputer__strategy': ['mean', 'median'],
                'logisticregression__C': [0.1, 0.5, 1.0],
                }
            grid_clf = GridSearchCV(model, param_grid, cv=10, iid=False)
            grid_clf.fit(X, y)
            self.best_params = grid_clf.best_params_
        else:
            model.fit(X, y)
        lr_model = model.named_steps['logisticregression']
        # average importances over classes
        importances = np.average(np.array(lr_model.coef_), axis=0)
        # average iterations over classes (can't take max_iter per class)
        iterations = np.average(lr_model.n_iter_, axis=0)

        # aggregate OHE feature importances, then check
        assert len(importances) == len(X_names)
        # model.named_steps['columntransformer'].transformers[1][1].get_feature_names(input_features=X_names)
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=iterations)

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        for col in X.names:
            XX = X[:, col]
            XX.replace(None, self.min[col])
            X[:, col] = XX
        model, _, _, _ = self.get_model_properties()
        X = X.to_pandas()
        if self.num_classes == 1:
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)
        return preds
