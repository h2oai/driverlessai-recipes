import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from h2oaicore.models import BaseCustomModel, LightGBMModel
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config, physical_cores_count

# Kaggle Problem: Amazon.com - Employee Access Challenge
# https://www.kaggle.com/c/amazon-employee-access-challenge

# Data: https://www.kaggle.com/c/amazon-employee-access-challenge/data

# Run DAI with 7/10/1 settings, AUC scorer, and exclude all models except for LIGHTGBMDEEP in expert settings -> custom recipes -> exclude specific models.

class MyLightGBMDeep(BaseCustomModel, LightGBMModel):
    _boosters = ['lightgbmdeep']
    _regression = False
    _binary = True
    _multiclass = False
    _display_name = "MYLGBMDEEP"
    _description = "LightGBM with more depth"
    _excluded_transformers = ['NumToCatWoETransformer','NumToCatWoEMonotonicTransformer','NumToCatTETransformer','NumCatTETransformer','OriginalTransformer','InteractionsTransformer','TruncSVDNumTransformer','ClusterTETransformer','ClusterIdTransformer','ClusterDistTransformer','CVCatNumEncode']

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        # First call the parent set_default_params
        LightGBMModel.set_default_params(
            self,
            accuracy=accuracy,
            time_tolerance=time_tolerance,
            interpretability=interpretability,
            **kwargs
        )
        # Then modify the parameters
        self.params["grow_policy"] = "lossguide"
        self.params["max_leaves"] = 8192

