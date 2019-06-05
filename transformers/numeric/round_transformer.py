from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyRoundTransformer(CustomTransformer):
    @staticmethod
    def get_parameter_choices():
        return {"decimals": [1, 2, 3]}

    @property
    def display_name(self):
        return "MyRound%dDecimals" % self.decimals

    def __init__(self, decimals, **kwargs):
        super().__init__(**kwargs)
        self.decimals = decimals

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return np.round(X.to_numpy(), decimals=self.decimals)
