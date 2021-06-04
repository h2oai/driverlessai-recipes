"""Modified version of Driverless AI's internal LightGBM implementation with a custom objective function (used for tree split finding).
"""
from h2oaicore.models import BaseCustomModel, LightGBMModel, XGBoostGBMModel
from h2oaicore.systemutils import config, loggerinfo
import numpy as np


class GBMTrainValidScoreGap:
    _regression = False  # target transformer not supported - only 1.10 LTS will support this feature natively
    _binary = True
    _multiclass = True
    _mojo = True
    _is_reproducible = True  # not be exactly reproducible on GPUs
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0, num_classes=None, **kwargs):
        max_abs_deviation = config.recipe_dict.get('max_abs_score_delta_train_valid', 0.0)  # set to > 0.0 to enable
        max_rel_deviation = config.recipe_dict.get('max_rel_score_delta_train_valid', 0.0)  # set to > 0.0 to enable
        return max_abs_deviation > 0 or max_rel_deviation > 0

    @staticmethod
    def do_acceptance_test():
        return False

    def post_fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        # determine the largest number of trees (from 1 to N, where N is what DAI would normally do) that

        # abs(training_score - valid_score) < abs-threshold IF abs-threshold > 0 ELSE true
        # AND
        # abs(training_score - valid_score) < rel-threshold * abs(training_score) IF rel-threshold > 0 ELSE true

        # To enable, set at least one of the two configurations by pasting the following (with modifications) into
        # "Add to config.toml via toml string" under Expert Settings -> Experiment:
        # #####
        # recipe_dict="{'max_rel_score_delta_train_valid': 0.1, 'max_abs_score_delta_train_valid': 0.01}"
        # #####
        max_abs_deviation = config.recipe_dict.get('max_abs_score_delta_train_valid', 0.0)  # set to > 0.0 to enable
        max_rel_deviation = config.recipe_dict.get('max_rel_score_delta_train_valid', 0.0)  # set to > 0.0 to enable
        logger = self.get_logger(**kwargs)
        if max_abs_deviation > 0 or max_rel_deviation > 0:
            if not (self._predict_by_iteration and eval_set and self.best_iterations):
                # LightGBM/XGB/CatBoost only
                return
            if "IS_SHIFT" in kwargs or "IS_LEAKAGE" in kwargs:
                # don't change leakage/shift detection logic
                return

            # goal is to find the new best_iterations, from 1...self.best_iterations
            max_n = max(self.best_iterations, 1)
            min_n = 0
            step_n = max(1, (max_n - min_n) // 20)  # try up to 20 steps from 1 to N trees

            mykwargs = {'output_margin': False, 'pred_contribs': False}
            self._predict_by_iteration = False  # allow override below
            self.model = self.get_model()
            valid_X = eval_set[0][0]
            valid_y = eval_set[0][1]
            valid_w = sample_weight_eval_set[0] if sample_weight_eval_set else None
            best_n = None
            best_train_score = None
            best_valid_score = None
            scorer = self.get_score_f()  # use the same scorer as the experiment
            for n in range(min_n, max_n, step_n):
                mykwargs[self._predict_iteration_name] = n  # fix number of trees for predict
                train_pred = self.predict_model_wrapper(X, **mykwargs)
                score_train = scorer(actual=y, predicted=train_pred, sample_weight=sample_weight,
                                     labels=self.labels)
                valid_pred = self.predict_model_wrapper(valid_X, **mykwargs)
                score_valid = scorer(actual=valid_y, predicted=valid_pred,
                                     sample_weight=valid_w, labels=self.labels)
                first_time = n == min_n
                abs_ok = max_abs_deviation <= 0 or \
                         np.abs(score_train - score_valid) <= max_abs_deviation
                rel_ok = max_rel_deviation <= 0 or \
                         np.abs(score_train - score_valid) <= max_rel_deviation * np.abs(score_train)
                if first_time or abs_ok and rel_ok:
                    # use the largest number n that satisfies this condition
                    best_n = n
                    best_train_score = score_train
                    best_valid_score = score_valid
                else:
                    # optimization: assume monotonic cross-over
                    break

            loggerinfo(logger,
                       "Changing optimal iterations from %d to %d to "
                       "keep train/valid %s gap below abs=%f, rel=%f: train: %f, valid: %f" %
                       (max_n, best_n, scorer.__self__.display_name,
                        max_abs_deviation, max_rel_deviation, best_train_score, best_valid_score))
            self._predict_by_iteration = True  # restore default behavior
            self.best_iterations = best_n  # update best iters <- this is the only effect of this method
        else:
            loggerinfo(logger,
                       "Train/valid gap control disabled - Must set at least one of the two settings to a value > 0.0, e.g.: "
                       "recipe_dict=\"{'max_rel_score_delta_train_valid': 0.1, 'max_abs_score_delta_train_valid': 0.01}\"")


class LightGBMTrainValidScoreGap(GBMTrainValidScoreGap, BaseCustomModel, LightGBMModel):
    pass


class XGBoostGBMTrainValidScoreGap(GBMTrainValidScoreGap, BaseCustomModel, XGBoostGBMModel):
    pass
