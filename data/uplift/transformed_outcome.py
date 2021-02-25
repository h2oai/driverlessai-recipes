from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
from datatable import f
import numpy as np
import pandas as pd
from h2oaicore.systemutils import config

class TransformedOutcome(CustomData):

    _treatment_col = config.recipe_dict['uplift.treatment_col'] if "uplift.treatment_col" in config.recipe_dict else "treatment"
    _outcome_col = config.recipe_dict['uplift.outcome_col'] if "uplift.outcome_col" in config.recipe_dict else "conversion"
    _transout_col = config.recipe_dict['uplift.transout_col'] if "uplift.transout_col" in config.recipe_dict else "transout"

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        if X is None or TransformedOutcome._treatment_col not in X.names or TransformedOutcome._treatment_col not in X.names:
            return []
        X['treatment_policy'] = X[:, TransformedOutcome._treatment_col].mean()
        X[TransformedOutcome._transout_col] = \
            X[:, f[TransformedOutcome._outcome_col] * ((f[TransformedOutcome._treatment_col] - f.treatment_policy) / (f.treatment_policy * (1 - f.treatment_policy)))]
        del X['treatment_policy']
        return X