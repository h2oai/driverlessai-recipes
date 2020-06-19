"""LightGBM/XGBoost with user-given monotonicity constraints (1/-1/0) for original numeric features
"""
from h2oaicore.models import BaseCustomModel, XGBoostGBMModel, LightGBMModel
from h2oaicore.transformers import Transformer
import numpy as np
import datatable as dt
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, config


class MonotonicGBMModel:
    """
    This recipe enables monotonicity constraints if the recipe_dict contains a valid feature->constraint mapping.

    Provide monotonicity constraints as additional config.toml value under Expert settings for Experiment, like so:
    recipe_dict = "{'monotonicity_constraints_dict': {'PAY_0': -1, 'PAY_2': -1, 'AGE': -1, 'BILL_AMT1': 1, 'PAY_AMT1': -1}}"

    Preview/AutoReport can show monotonicity constraints as disabled, but you can ignore that, as recipe takes over.
    """
    _multiclass = False  # not supported
    _can_handle_categorical = False
    _is_reproducible = False  # not reproducible on GPUs
    _included_transformers = ['OriginalTransformer']  # want monotonicity on orig features, disable feature engineering

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

        # disable default handling of monotonicity constraints in DAI
        self.params["monotonicity_constraints"] = False

        # set custom monotonicity constraints, or fall back to 0 if not provided
        constraints = [str(user_constraints.get(x, 0)) for x in X_names_raw]
        self.set_constraints(constraints)

        # optional logging
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)
            loggerinfo(logger, "Monotonicity constraints: %s" % {x: user_constraints.get(x, 0) for x in X_names_raw})


class MonotonicXGBoostModel(MonotonicGBMModel, XGBoostGBMModel, BaseCustomModel):
    _description = "XGBoostGBM with user-given monotonicity constraints on original numeric features"

    def set_constraints(self, constraints):
        self.params['monotone_constraints'] = "(" + ",".join(constraints) + ")"


class MonotonicLightGBMModel(MonotonicGBMModel, LightGBMModel, BaseCustomModel):
    _description = "LightGBM with user-given monotonicity constraints on original numeric features"

    def set_constraints(self, constraints):
        self.lightgbm_params['monotone_constraints'] = ",".join(constraints)

