"""Modify dataset with arbitrary env"""

from h2oaicore.data import CustomData


class FreshEnvData(CustomData):
    # Specify the python package dependencies.  Will be installed in order of list
    # Below caches the env into "id" folder
    #isolate_env = dict(pyversion="3.6", install_h2oaicore=False, install_datatable=True, modules_needed_by_name=["pandas==1.1.5"], cache_env=True, file=__file__, id="myrecipe12345")
    # Below does not cache the env
    isolate_env = dict(pyversion="3.8", install_h2oaicore=False, install_datatable=True, modules_needed_by_name=["pandas==1.1.5"])
    @staticmethod
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
