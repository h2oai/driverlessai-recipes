"""A best effort transformer to determine browser device characteristics from a user-agent string"""
#
# Custom transformer: UserAgent
# UserAgent column should have one of the following names: ua, user-agent, user_agent, useragent
#

import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomTransformer


class UserAgent(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

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
        def get_ua_info(ua_string):
            from user_agents import parse
            ua = parse(ua_string)
            return ua.browser.family, ua.os.family, ua.device.family, ua.is_mobile, ua.is_tablet

        ua_column_names = ['ua', 'user-agent', 'user_agent', 'useragent']
        col_name = X.names[0]
        if col_name in ua_column_names:
            newnames = ("browser", "os", "device", "is_mobile", "is_tablet")
            Y = X[col_name].to_list()[0]
            Z = dt.Frame([get_ua_info(x) for x in Y], names=[f"{col_name}_{s}" for s in newnames])
            X.cbind(Z)
            return X
        else:
            return X.to_pandas().iloc[:, 0]
