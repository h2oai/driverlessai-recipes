"""XGBoost for uplift modeling"""

"""Modified version of Driverless AI's internal XGB implementation with transformed outcome transformation.
See e.g. https://pylift.readthedocs.io/en/latest/introduction.html#the-transformed-outcome for more information.

This recipe is intended to be used for creating binary classification uplift models using the transformed outcome approach.
You need to use one of the transformed outcome uplift models along with the AUUC scorer. The `treatment` column is passed
as a `weight` and the Driverless' task needs to be _regression_ (since after the transformation the target variable
becomes the uplift directly).
"""
from h2oaicore.models import BaseCustomModel, XGBoostGBMModel
import numpy as np
import datatable as dt


class XGBoostGBMModelTransformedOutcome(BaseCustomModel, XGBoostGBMModel):
    _regression = True
    _binary = False
    _multiclass = False
    _mojo = True
    _is_reproducible = False  # might not reproduce identically on GPUs
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _description = "Transformed Outcome Uplift Estimator based on XGBoost"
    _display_name = "XGBoostTO"

    def fit(self, X: dt.Frame, y: np.array, sample_weight: np.array = None,
            eval_set=None, sample_weight_eval_set=None, **kwargs):
        if sample_weight is not None:
            treatment_policy = np.mean(sample_weight)  # weights are carrying the treatment
            y = y * ((sample_weight - treatment_policy) / (treatment_policy * (1 - treatment_policy)))
        return super().fit(X, y, None, eval_set, None, **kwargs)

    @staticmethod
    def do_acceptance_test():
        return False
