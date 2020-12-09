"""Modified version of Driverless AI's internal LightGBM implementation with for quantile regression
"""
from h2oaicore.models import BaseCustomModel, LightGBMModel
import numpy as np
from h2oaicore.systemutils import config
import datatable as dt

class LightGBMTransformedOutcomeEstimator(BaseCustomModel, LightGBMModel):
    _regression = True
    _binary = False
    _multiclass = False
    _mojo = True
    _is_reproducible = False  # might not reproduce identically on GPUs
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _description = "Transformed Outcome Uplift Estimator based on LightGBM"
    _display_name = "TO LightGBM Estimator"

    def fit(self, X: dt.Frame, y: np.array, sample_weight: np.array = None,
            eval_set=None, sample_weight_eval_set=None, **kwargs):
        treatment_policy = np.mean(sample_weight)  # weights are carrying the treatment
        y = y * ((sample_weight - treatment_policy) / (treatment_policy * (1 - treatment_policy)))
        return super().fit(X, y, None, eval_set, None, **kwargs)


    @staticmethod
    def do_acceptance_test():
        return False
