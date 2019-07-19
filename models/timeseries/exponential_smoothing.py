"""Linear Model on top of Exponential Weighted Moving Average Lags for Time-Series. 
Provide appropriate lags and past outcomes during batch scoring for best results."""
from h2oaicore.models import BaseCustomModel, GLMModel
import numpy as np


class ExponentialSmoothingModel(BaseCustomModel, GLMModel):
    _regression = True
    _binary = False
    _multiclass = False
    _time_series_only = True
    _display_name = "EWMA_GLM"
    _description = "GLM with EWMA Lags"
    _included_transformers = ["EwmaLagsTransformer"]

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0):
        return True  # i.e. ignore GLM's restrictions on interpretability and accuracy

    @staticmethod
    def enabled_setting():
        return "on"  # i.e. ignore GLM's default choice of auto that would disable this model if too many classes
