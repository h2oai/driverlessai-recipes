"""For GPU usage testing purposes."""

import os
import numpy as np
from h2oaicore.models import CustomModel


class CustomGpuCheck(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False  # WIP

    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _get_gpu_lock = True  # whether to lock GPUs for this model before fit and predict
    _must_use_gpu = True  # this recipe can only be used if have GPUs
    _predict_on_same_gpus_as_fit = True  # force predict to behave like fit, regardless of config.num_gpus_for_prediction

    @staticmethod
    def do_acceptance_test():
        return False

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        self.params = {}

    def mutate_params(self,
                      **kwargs):
        self.params = {}

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        try:
            x = os.environ['CUDA_VISIBLE_DEVICES']
            if x == '':
                raise AssertionError(f'CUDA_VISIBLE_DEVICES = {x} should not be set.')
        except KeyError:
            pass

        self.set_model_properties(model=[1],
                                  features=list(X.names),
                                  importances=([1.0] * len(list(X.names))),
                                  iterations=0)

    def predict(self, X, **kwargs):
        """
        Returns: dt.Frame, np.ndarray or pd.DataFrame, containing predictions (target values or class probabilities)
        Shape: (K, c) where c = 1 for regression or binary classification, and c>=3 for multi-class problems.
        """
        try:
            x = os.environ['CUDA_VISIBLE_DEVICES']
            if x == '':
                raise AssertionError(f'CUDA_VISIBLE_DEVICES = {x} should not be set.')
        except KeyError:
            pass

        return np.random.randint(0, 2, (X.nrows, 1))


# Not sure if we need the same model again, blending may work with only one model type, too.
class CustomGpuCheck2(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False  # WIP

    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _get_gpu_lock = True  # whether to lock GPUs for this model before fit and predict
    _must_use_gpu = True  # this recipe can only be used if have GPUs
    _predict_on_same_gpus_as_fit = True  # force predict to behave like fit, regardless of config.num_gpus_for_prediction

    @staticmethod
    def do_acceptance_test():
        return False

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        self.params = {}

    def mutate_params(self,
                      **kwargs):
        self.params = {}

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        try:
            x = os.environ['CUDA_VISIBLE_DEVICES']
            if x == '':
                raise AssertionError(f'CUDA_VISIBLE_DEVICES = {x} should not be set.')
        except KeyError:
            pass

        self.set_model_properties(model=[1],
                                  features=list(X.names),
                                  importances=([1.0] * len(list(X.names))),
                                  iterations=0)

    def predict(self, X, **kwargs):
        """
        Returns: dt.Frame, np.ndarray or pd.DataFrame, containing predictions (target values or class probabilities)
        Shape: (K, c) where c = 1 for regression or binary classification, and c>=3 for multi-class problems.
        """
        try:
            x = os.environ['CUDA_VISIBLE_DEVICES']
            if x == '':
                raise AssertionError(f'CUDA_VISIBLE_DEVICES = {x} should not be set.')
        except KeyError:
            pass

        return np.random.randint(0, 2, (X.nrows, 1))
