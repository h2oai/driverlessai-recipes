"""Extract column data using grok patterns"""

import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomTransformer

#
# Extract data from a composite column record using grok mapping a la logstash
#
# Example pattern:
_PATTERN = '%{TIMESTAMP_ISO8601:ts} %{IPV4:ip} %{NUMBER:status:int}: %{LOGLEVEL:level} %{WORD:class} %{DATA}'
#
# Column name to transform using the pattern above
_COLUMN_TO_PARSE = 'syslog'
#
#
# These new columns will be added based on the pattern above:
# syslog_ts, syslog_ip, syslog_status, syslog_level, syslog_class
#
# CSV example:
#
# column1,syslog,some_result
# hello,"2011-09-01 11:00:33.4444 10.10.10.1 200: INFO class1 Some data here 1",0
# hello,"2011-09-02 12:00:33.4444 10.10.10.1 200: WARN class2 Some data here 2",0
#


class TextGrokParser(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _modules_needed_by_name = ["pygrok==1.0.0"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from pygrok import Grok
        self.grok = Grok(_PATTERN)
        self.columns = list(self.grok.regex_obj.groupindex.keys())
        self.column_to_parse = _COLUMN_TO_PARSE

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def do_acceptance_test():
        return False

    # noinspection PyBroadException
    def parse_text(self, text):
        try:
            x = self.grok.match(text)
            return tuple(x[k] for k in self.columns)
        except Exception:
            return tuple(None for _ in self.columns)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        col_name = X.names[0]
        if col_name == self.column_to_parse:
            Y = X[col_name].to_list()[0]
            Z = dt.Frame([self.parse_text(x) for x in Y], names=[f"{col_name}_{s}" for s in self.columns])
            return Z
        else:
            return X.to_pandas().iloc[:, None]
