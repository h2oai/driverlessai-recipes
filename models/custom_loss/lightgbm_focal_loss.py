"""Modified version of Driverless AI's internal LightGBM implementation with a custom objective function (used for tree split finding).
"""
from h2oaicore.models import BaseCustomModel, LightGBMModel
import numpy as np
from scipy.misc import derivative


def focal_loss(y_pred, y_true):
    a, g = 0.25, 1

    def fl(x, t):
        p = 1 / (1 + np.exp(-x))
        return -(a * t + (1 - a) * (1 - t)) * ((1 - (t * p + (1 - t) * (1 - p))) ** g) * (
                t * np.log(p) + (1 - t) * np.log(1 - p))

    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess


def focal_loss_eval_error(y_pred, y_true):
    a, g = 0.25, 1
    p = 1 / (1 + np.exp(-y_pred))
    loss = -(a * y_true + (1 - a) * (1 - y_true)) * ((1 - (y_true * p + (1 - y_true) * (1 - p))) ** g) * (
                y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return 'focal_loss', np.mean(loss), False


class MyLGBMFocalLoss(BaseCustomModel, LightGBMModel):
    """Custom model class that re-uses DAI LightGBMModel
    The class inherits :
      - BaseCustomModel that really is just a tag. It's there to make sure DAI knows it's a custom model and not
      its inner LightGBM Model
      - LightGBMModel object so that the custom model inherits all the properties and methods, especially for params
      mutation
    """
    _regression = False
    _binary = True
    _multiclass = False
    _mojo = True
    # Give the display name and description that will be shown in the UI
    _display_name = "MyLGBMFocalLoss"
    _description = "LightGBM with custom focal loss"
    _is_reproducible = False  # might not be exactly reproducible on GPUs
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        # Define the global loss
        # global custom_asymmetric_objective

        # First call the LightGBM set_default_params
        # This will input all model parameters just like DAI would do.
        LightGBMModel.set_default_params(
            self,
            accuracy=accuracy,
            time_tolerance=time_tolerance,
            interpretability=interpretability,
            **kwargs
        )
        # Now we just need to tell LightGBM that it has to optimize for our custom objective
        # And we are done
        self.params["objective"] = focal_loss
        self.params["eval_metric"] = focal_loss_eval_error

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
        # Now set the objective
        self.params["objective"] = focal_loss
        self.params["eval_metric"] = focal_loss_eval_error

    def predict(self, X, y=None, **kwargs):
        from h2oaicore.models import sigmoid
        preds = super().predict(X, **kwargs)
        if not kwargs.get('pred_contribs', False) and not kwargs.get('output_margin', False):
            preds = sigmoid(preds)
        return preds
