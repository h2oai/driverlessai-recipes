"""Modified version of Driverless AI's internal LightGBM implementation with a custom objective function (used for tree split finding).
"""
from h2oaicore.models import BaseCustomModel, LightGBMModel
from h2oaicore.metrics import LogLossScorer, RmseScorer
import numpy as np


class LightGBMTrainValidLossGap(BaseCustomModel, LightGBMModel):
    """Custom model class that re-uses DAI LightGBMModel
    The class inherits :
      - BaseCustomModel that really is just a tag. It's there to make sure DAI knows it's a custom model and not
      its inner LightGBM Model
      - LightGBMModel object so that the custom model inherits all the properties and methods, especially for params
      mutation
    """
    _regression = True
    _binary = True
    _multiclass = True
    _mojo = True
    # Give the display name and description that will be shown in the UI
    _display_name = "LGBMTrainValidLossGap"
    _description = "LightGBM with custom early stopping based on difference between train and valid loss"
    _is_reproducible = True  # not be exactly reproducible on GPUs
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def post_fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        # determine the largest number of trees (from 1 to N, where N is what DAI would normally do) that satisfies the condition that
        # abs(training_score - valid_score) < user-given-threshold
        # as configured below


        # customize - start
        scorer = LogLossScorer() if self.num_classes >= 2 else RmseScorer()  # other choices: AucScorer, F1Scorer, etc.
        max_abs_deviation = 0.02  # stop as soon as loss for train (not holdout) and valid deviate by more than this value
        # customize - end
            
        if eval_set and self.best_iterations and self._predict_by_iteration:
            if "IS_SHIFT" in kwargs or "IS_LEAKAGE" in kwargs:
                # don't change leakage/shift detection logic
                return

            # goal is to find the new best_iterations, from 1...self.best_iterations
            max_n = self.best_iterations
            min_n = 1
            step_n = max(1, (max_n - min_n) // 20)  # try up to 20 steps from 1 to N trees

            best_n = min_n
            mykwargs = {}
            mykwargs['output_margin'] = False
            mykwargs['pred_contribs'] = False
            self._predict_by_iteration = False  # allow override below
            self.model = self.get_model()
            valid_X = eval_set[0][0]
            valid_y = eval_set[0][1]
            valid_w = sample_weight_eval_set[0] if sample_weight_eval_set else None
            for n in range(min_n, max_n, step_n):
                mykwargs[self._predict_iteration_name] = n  # fix number of trees for predict
                train_pred = self.predict_model_wrapper(X, **mykwargs)
                score_train = scorer.score_base(actual=y, predicted=train_pred, sample_weight=sample_weight, labels=self.labels)
                valid_pred = self.predict_model_wrapper(valid_X, **mykwargs)
                score_valid = scorer.score_base(actual=valid_y, predicted=valid_pred,
                                                sample_weight=valid_w, labels=self.labels)
                if np.abs(score_train - score_valid) < max_abs_deviation:
                    # use the largest number n that satisfies this condition
                    best_n = n
                else:
                    # optimization: assume monotonic cross-over
                    break 

            print("Changing optimal iterations from %d to %d to keep train/valid %s loss gap below %f" %
                        (max_n, best_n, scorer.display_name, max_abs_deviation))
            self.best_iterations = best_n
            self._predict_by_iteration = True  # restore default behavior
