"""Template base class for a custom data recipe."""

import datatable as dt
import numpy as np
import pandas as pd

class CustomData(BaseData):
    """Base class for a custom data creation recipe that can be specified externally to Driverless AI.

    Note: Experimental API, will most likely change in future versions.
    """
    @staticmethod
    def create_data(X: dt.Frame=None) -> Union[str, List[str],
                                               dt.Frame, List[dt.Frame],
                                               np.ndarray, List[np.ndarray],
                                               pd.DataFrame, List[pd.DataFrame]]:
        """
        Create one or multiple datasets to be imported into Driverless AI

        Args:
        X (:obj:`dt.Frame`): `Python datatable github<https://github.com/h2oai/datatable>`
        If provided, this data creation recipe can start with an existing dataset from Driverless AI.
        Otherwise, have to create the data from scratch.
        """
        return X

