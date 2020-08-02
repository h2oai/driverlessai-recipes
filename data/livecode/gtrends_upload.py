import datatable as dt
import numpy as np
import pandas as pd
from typing import Union, List, Dict
from h2oaicore.data import BaseData

_global_modules_needed_by_name = ["pytrends==4.6.0"]

class CustomData(BaseData):

    from pytrends.request import TrendReq

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

        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = ["sanitizing wipes"]
        geo = ["US-TX", "US-CA", "US-NY"]
        geo = "US-TX"
        timeframe = '2020-01-01 2020-05-08'
        timeframe = 'today 5-y'
        pytrends.build_payload(kw_list, timeframe=timeframe, geo=geo)

        trends = pytrends.interest_over_time()
        X = dt.Frame(date = trends.index.to_list())
        X.cbind(dt.Frame(gtrend = trends.iloc[:, 0].tolist(), isPartial = trends.iloc[:, 1].tolist()))

        return {"gtrends_sanitizing_wipes":X[dt.f.isPartial == 'False', :]}
