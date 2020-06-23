"""LightGBM/XGBoostGBM/DecisionTree with user-given monotonicity constraints (1/-1/0) for original numeric features
"""
from h2oaicore.models import BaseCustomModel, XGBoostGBMModel, LightGBMModel, DecisionTreeModel
from h2oaicore.transformers import Transformer
import numpy as np
import datatable as dt
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, config, loggerdata


class MonotonicModel:
    """
    This recipe forces user-defined monotonicity constraints on *original* numeric features. It is "just" a convenience wrapper to disable feature engineering, and to only show up if user-specified constraints are active.

    Add monotonicity constraints under Expert Settings -> Features -> Manual override for monotonicity constraints.
    E.g., monotonicity_constraints_dict = {'PAY_0': -1, 'PAY_2': -1, 'AGE': -1, 'BILL_AMT1': 1, 'PAY_AMT1': -1}
    """
    _multiclass = False  # not supported
    _can_handle_categorical = False
    _included_transformers = ['OriginalTransformer']  # want monotonicity on orig features, disable feature engineering

    @staticmethod
    def do_acceptance_test():
        return False  # no need

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        # only enable this model if the user provides custom monotonicity constraints.
        return len(config.monotonicity_constraints_dict) > 0


# https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
# only supported for DAI 1.8.7.1+
class MonotonicXGBoostModel(MonotonicModel, XGBoostGBMModel, BaseCustomModel):
    _description = "XGBoostGBM with user-given monotonicity constraints on original numeric features"


# https://lightgbm.readthedocs.io/en/latest/Parameters.html#monotone_constraints
# only supported for DAI 1.8.7.1+
class MonotonicLightGBMModel(MonotonicModel, LightGBMModel, BaseCustomModel):
    _description = "LightGBM with user-given monotonicity constraints on original numeric features"


# https://lightgbm.readthedocs.io/en/latest/Parameters.html#monotone_constraints
# only supported for DAI 1.8.7.1+
class MonotonicDecisionTreeModel(MonotonicModel, DecisionTreeModel, BaseCustomModel):
    _description = "DecisionTree with user-given monotonicity constraints on original numeric features"

