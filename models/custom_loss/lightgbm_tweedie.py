"""Modified version of Driverless AI's internal LightGBM implementation with tweedie distribution
"""
from h2oaicore.models import BaseCustomModel, LightGBMModel
import numpy as np


class TweedieLightGBMModel(BaseCustomModel, LightGBMModel):
    _regression = True
    _binary = False
    _multiclass = False
    _mojo = True
    _is_reproducible = False  # might not reproduce identically on GPUs
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _tweedie_variance_power = 1.5  # PLEASE CONFIGURE

    _description = "LightGBM with Tweedie distribution with tweedie variance power=%g" % _tweedie_variance_power
    _display_name = "LightGBM tweedie variance power=%g" % _tweedie_variance_power

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        # First call the LightGBM set_default_params
        # This will input all model parameters just like DAI would do.
        LightGBMModel.set_default_params(
            self,
            accuracy=accuracy,
            time_tolerance=time_tolerance,
            interpretability=interpretability,
            **kwargs
        )
        # Now we just need to tell LightGBM to use tweedie distribution
        self.params["objective"] = "tweedie"
        self.params["tweedie_variance_power"] = TweedieLightGBMModel._tweedie_variance_power

    def mutate_params(
            self, get_best=False, time_tolerance=10, accuracy=10, interpretability=1,
            imbalance_ratio=1.0,
            train_shape=(1, 1), ncol_effective=1,
            time_series=False, ensemble_level=0,
            score_f_name: str = None, **kwargs):
        # If we don't override the parent mutate_params method, DAI would have the opportunity
        # to modify the objective and select the winner
        # For demonstration purposes we purposely make sure that the objective
        # is the one we want
        # So first call the parent method to mutate parameters
        super().mutate_params(
            get_best=get_best, time_tolerance=time_tolerance, accuracy=accuracy,
            interpretability=interpretability,
            imbalance_ratio=imbalance_ratio, train_shape=train_shape, ncol_effective=ncol_effective,
            time_series=time_series, ensemble_level=ensemble_level,
            score_f_name=score_f_name, **kwargs)
        # Now we just need to tell LightGBM to use tweedie distribution
        self.params["objective"] = "tweedie"
        self.params["tweedie_variance_power"] = TweedieLightGBMModel._tweedie_variance_power