"""datatable Linear Model"""
from h2oaicore.models import CustomModel
from datatable import dt, models, f
import math
import numpy as np


class datatableLinearModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "datatableLinearModel"
    _description = "datatable Linear Model"

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        if self.num_classes == 1:
            model_type = "regression"
        elif self.num_classes == 2:
            model_type = "binomial"
        else:
            model_type = "multinomial"

        lm = dt.models.LinearModel(model_type=model_type)
        y = dt.Frame(y)
        X_mean = X.mean()
        X_sd = X.sd()
        # for constant columns prevent dividing by zero
        X_sd.replace(0.0, 1.0)
        X_standard = X[:, (f[:] - X_mean) / X_sd]
        self.impute(X_standard)

        res = lm.fit(X_standard, y)
        importances = lm.model[1:, dt.rowsum(dt.math.abs(f[:]))]
        model = {"lm": lm, "X_mean": X_mean, "X_sd": X_sd}

        self.set_model_properties(model=model,
                                  features=X.names,
                                  importances=importances.to_list()[0],
                                  iterations=res.epoch)

    def predict(self, X, **kwargs):
        model, _, _, _ = self.get_model_properties()
        X_standard = X[:, (f[:] - model["X_mean"]) / model["X_sd"]]
        self.impute(X_standard)
        p = model["lm"].predict(X_standard)
        return p

    @staticmethod
    def impute(X):
        X.replace([None, -math.inf, math.inf], 0.0)
