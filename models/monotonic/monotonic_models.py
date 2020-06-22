"""LightGBM/XGBoostGBM/DecisionTree with user-given monotonicity constraints (1/-1/0) for original numeric features
"""
from h2oaicore.models import BaseCustomModel, XGBoostGBMModel, LightGBMModel, DecisionTreeModel
from h2oaicore.transformers import Transformer
import numpy as np
import datatable as dt
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, config, loggerdata


class MonotonicGBMModel:
    """
    This recipe enables monotonicity constraints if the recipe_dict contains a valid feature->constraint mapping.

    Provide monotonicity constraints as additional config.toml value under Expert settings for Experiment, like so:
    recipe_dict = "{'monotonicity_constraints_dict': {'PAY_0': -1, 'PAY_2': -1, 'AGE': -1, 'BILL_AMT1': 1, 'PAY_AMT1': -1}}"

    Preview/AutoReport can show monotonicity constraints as disabled, but you can ignore that, as recipe takes over.
    """
    _multiclass = False  # not supported
    _can_handle_categorical = False
    _included_transformers = ['OriginalTransformer']  # want monotonicity on orig features, disable feature engineering

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        # only enable this model if interpretability setting is high enough (>= 7 by default)
        return interpretability >= config.monotonicity_constraints_interpretability_switch

    # this method is called before fit(), and we use this opportunity to set some internal dicts
    def pre_fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        super().pre_fit(X, y, sample_weight=sample_weight, eval_set=eval_set,
                        sample_weight_eval_set=sample_weight_eval_set, **kwargs)  # usually no-op, but can change later

        # get original column names (can be subset of training data columns used for experiment)
        X_names_raw = list(dt.Frame(X).names)
        if not kwargs.get("IS_LEAKAGE", False) and not kwargs.get("IS_SHIFT", False):
            # for leakage/shift, already get raw feature names without gene_id prefix
            X_names_raw = [Transformer.raw_feat_name(c) for c in X_names_raw]

        # read user-defined monotonicity constraints, or empty dict
        user_constraints = config.recipe_dict.get('monotonicity_constraints_dict', {})
        # sanity checks
        assert all(x == 1 or x == 0 or x == -1 for x in user_constraints.values()), \
            "monotonicity_constraints_dict must contain only values of 0, 1 or -1"

        # disable default handling of monotonicity constraints in DAI <= 1.8.7
        self.params["monotonicity_constraints"] = False
        # set custom monotonicity constraints, or fall back to 0 if not provided
        constraints = [user_constraints.get(x, 0) for x in X_names_raw]
        self.set_constraints(constraints)

        # optional logging
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)
            loggerdata(logger, "Monotonicity constraints: %s" % {x: user_constraints.get(x, 0) for x in X_names_raw})


# https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
class MonotonicXGBoostModel(MonotonicGBMModel, XGBoostGBMModel, BaseCustomModel):
    _description = "XGBoostGBM with user-given monotonicity constraints on original numeric features"
    _can_use_gpu = False  # faster and more reproducible on CPU

    def set_constraints(self, constraints):
        self.params['monotone_constraints'] = '(' + ",".join([str(x) for x in constraints]) + ")"


# https://lightgbm.readthedocs.io/en/latest/Parameters.html#monotone_constraints
# only supported for DAI 1.8.7.1+
class MonotonicLightGBMModel(MonotonicGBMModel, LightGBMModel, BaseCustomModel):
    _description = "LightGBM with user-given monotonicity constraints on original numeric features"
    _can_use_gpu = False  # faster and more reproducible on CPU

    def set_constraints(self, constraints):
        self.lightgbm_params['monotone_constraints'] = constraints
        self.lightgbm_params['monotone_penalty'] = 20  # greater than max depth


# https://lightgbm.readthedocs.io/en/latest/Parameters.html#monotone_constraints
# only supported for DAI 1.8.7.1+
class MonotonicDecisionTreeModel(MonotonicGBMModel, DecisionTreeModel, BaseCustomModel):
    _description = "DecisionTree with user-given monotonicity constraints on original numeric features"
    _can_use_gpu = False  # faster and more reproducible on CPU

    def set_constraints(self, constraints):
        self.lightgbm_params['monotone_constraints'] = constraints
        self.lightgbm_params['monotone_penalty'] = 20  # greater than max depth

