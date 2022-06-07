"""Modify dataset with arbitrary env"""

from h2oaicore.data import CustomData
from h2oaicore.utils import wrap_create


class FreshEnvData(CustomData):
    @staticmethod
    # Specify the python package dependencies.  Will be installed in order of list
    # NOTE: Keep @wrap_create on a single line
    # NOTE: If want to share cache across recipes, can set cache_env=True and set id=<some unique identifier, like myrecipe12345>
    # Below caches the env into "id" folder
    # @wrap_create(pyversion="3.6", install_h2oaicore=False, install_datatable=True, modules_needed_by_name=["pandas==1.1.5"], cache_env=True, file=__file__, id="myrecipe12345")
    # Below does not cache the env
    @wrap_create(pyversion="3.8", install_h2oaicore=False, install_datatable=True,
                 modules_needed_by_name=["pandas==1.1.5"], file=__file__)
    def create_data(X=None):
        import os
        import datatable as dt
        if X is not None and os.path.isfile(X):
            X = dt.fread(X)
        else:
            X = None

        my_path = os.path.dirname(__file__)

        import pandas as pd
        assert pd.__version__ == "1.1.5", "actual: %s" % pd.__version__

        url = "http://data.un.org/_Docs/SYB/CSV/SYB63_226_202009_Net%20Disbursements%20from%20Official%20ODA%20to%20Recipients.csv"
        import urllib.request

        new_file = os.path.join(my_path, "user_file.csv")
        urllib.request.urlretrieve(url, new_file)

        import datatable as dt
        return dt.fread(new_file)
