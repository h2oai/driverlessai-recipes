"""Linear Model on top of Exponential Weighted Moving Average Lags for Time-Series.
Provide appropriate lags and past outcomes during batch scoring for best results."""
from h2oaicore.models import BaseCustomModel, GLMModel
import numpy as np


class ExponentialSmoothingModel(BaseCustomModel, GLMModel):
    _regression = True
    _binary = False
    _multiclass = False
    _display_name = "EWMA_GLM"
    _description = "GLM with EWMA Lags"
    _included_transformers = ["EwmaLagsTransformer"]

