"""A best effort to determine a browser device characteristics from a user-agent string"""
#
# Custom transformer: UserAgent
# UserAgent column should have one fo the following names: ua, user-agent, user_agent, useragent
#

import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomTransformer
from user_agents import parse


def get_ua_info(ua_string):
    ua = parse(ua_string)
    return ua.browser.family, ua.os.family, ua.device.family, ua.is_mobile, ua.is_tablet


class UserAgent(CustomTransformer):

    _modules_needed_by_name = ["user-agents==2.0"]

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def do_acceptance_test():
        return False

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        ua_column_names = ['ua', 'user-agent', 'user_agent', 'useragent']
        col_name = X.names[0]
        if col_name in ua_column_names:
            _X = X.to_pandas()
            _X[col_name + "_browser"] = _X[col_name].apply(lambda x: get_ua_info(x)[0])
            _X[col_name + "_os"] = _X[col_name].apply(lambda x: get_ua_info(x)[1])
            _X[col_name + "_device"] = _X[col_name].apply(lambda x: get_ua_info(x)[2])
            _X[col_name + "_is_mobile"] = _X[col_name].apply(lambda x: get_ua_info(x)[3])
            _X[col_name + "_is_tablet"] = _X[col_name].apply(lambda x: get_ua_info(x)[4])
            dt.DataTable(_X)
        else:
            return X.to_pandas().iloc[:, 0]
