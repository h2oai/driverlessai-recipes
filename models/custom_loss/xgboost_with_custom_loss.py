"""Modified version of Driverless AI's internal XGBoost implementation with a custom objective function (used for tree split finding).
"""
from h2oaicore.models import BaseCustomModel, XGBoostGBMModel
import numpy as np


# The custom objective function will be pickled along with the underlying XGBoost model for persistance purposes
# as a result it can't a lambda function or a method of the custom model object
# The only option is to make the function global in the following manner
def custom_asymmetric_objective(y_true, y_pred):
    """Asymetric MSE loss
    A custom loss has to return the gradient and the hessian of the loss or objective function
    This is not an evaluation but the loss LightGBM will optimize during training
    """
    residual = (y_true - y_pred).astype("float")
    grad = np.where(residual < 0, -2 * 10.0 * residual, -2 * residual)
    hess = np.where(residual < 0, 2 * 10.0, 2.0)
    return grad, hess


class MyXGBMAsymMSE(BaseCustomModel, XGBoostGBMModel):
    """Custom model class that re-uses DAI XGBoostGBMModel
    The class inherits :
      - BaseCustomModel that really is just a tag. It's there to make sure DAI knows it's a custom model and not
      its inner XGBoost Model
      - XGBoostGBMModel object so that the custom model inherits all the properties and methods, especially for params
      mutation
    """
    # The loss is a regression loss
    _regression = True
    # The loss is not for classification
    _binary = False
    _multiclass = False  # WIP
    # Give the display name and description that will be shown in the UI
    _display_name = "MYXGBMAsymMSE"
    _description = "XGBoost with custom asymetric loss/objective"
    _boosters = ["xgbasym"]

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        # First call the XGBoostGBMModel set_default_params
        # This will input all model parameters just like DAI would do.
        XGBoostGBMModel.set_default_params(
            self,
            accuracy=accuracy,
            time_tolerance=time_tolerance,
            interpretability=interpretability,
            **kwargs
        )
        # Now we just need to tell XGBoost that it has to optimize for our custom objective
        # And we are done
        self.params["objective"] = custom_asymmetric_objective

    def mutate_params(self, get_best=False, time_tolerance=None, accuracy=None, interpretability=None,
                      imbalance_ratio=None,
                      train_shape=None, ncol_effective=None,
                      time_series=False, ensemble_level=None,
                      score_f_name: str = None, **kwargs):
        # If we don't override the parent mutate_params method, DAI would have the opportunity
        # to modify the objective and select the winner
        # For demonstration purposes we purposely make sure that the objective
        # is the one we want
        # So first call the parent method to mutate parameters
        params = XGBoostGBMModel.mutate_params(
            self, get_best=get_best, time_tolerance=time_tolerance, accuracy=accuracy,
            interpretability=interpretability,
            imbalance_ratio=imbalance_ratio,
            train_shape=train_shape, ncol_effective=ncol_effective,
            time_series=time_series, ensemble_level=ensemble_level,
            score_f_name=score_f_name, **kwargs
        )
        # Now set the objective that DAI could have mutated
        params["objective"] = custom_asymmetric_objective
