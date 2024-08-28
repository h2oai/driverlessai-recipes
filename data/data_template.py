"""Custom data recipe base class"""

import datatable as dt
import numpy as np
import pandas as pd
from typing import Union, List, Dict
from h2oaicore.data import BaseData

_global_modules_needed_by_name = []  # Optional global package requirements, for multiple custom recipes in a file

class CustomData(BaseData):
    """Base class for a custom data creation recipe that can be specified externally to Driverless AI.
    To use as recipe, in the class replace CustomData with your class name and replace BaseData with CustomData

    Note: Experimental API, will most likely change in future versions.
    """

    """Specify the python package dependencies (will be installed via pip install mypackage==1.3.37)"""
    _modules_needed_by_name = []  # List[str], e.g., ["mypackage==1.3.37"]

    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[
        str, List[str],
        dt.Frame, List[dt.Frame],
        np.ndarray, List[np.ndarray],
        pd.DataFrame, List[pd.DataFrame],
        Dict[str, str],  # {data set names : paths}
        Dict[str, dt.Frame],  # {data set names : dt frames}
        Dict[str, np.ndarray],  # {data set names : np arrays}
        Dict[str, pd.DataFrame],  # {data set names : pd frames}
    ]:
        """
        Create one or multiple datasets to be imported into Driverless AI

        Args:
        X (:obj:`dt.Frame`): `Python datatable github<https://github.com/h2oai/datatable>`
        If provided, this data creation recipe can start with an existing dataset from Driverless AI.
        Otherwise, have to create the data from scratch.
        """
        # If data recipes takes in and uses X,
        # the below ensures upload-time (when data not provided to recipe) will not fail
        # The below also can be used to indicate if data recipe is just setup without all details required,
        # like a path to files or a link that needs to be filled in.  Then one can use the condition that the path
        # exists to run, and if no path, then just return [] so does not fail CI testing, etc.
        if X is None:
            return []

        return X

"""Example custom data recipe."""

from h2oaicore.data import CustomData
import datatable as dt
import numpy as np

class ExampleCustomData(CustomData):
    @staticmethod
    def create_data():
        return dt.Frame(np.array([[1,2,3],[4,5,6]]))


