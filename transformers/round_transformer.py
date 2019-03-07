from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyRound1DigitTransformer(CustomTransformer):
    def __init__(self, decimals=1, **kwargs):
        super().__init__(**kwargs)
        self.decimals = decimals

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return np.round(X.to_numpy(), decimals=self.decimals)


class MyRound2DigitsTransformer(MyRound1DigitTransformer):
    def __init__(self, decimals=2, **kwargs):
        super().__init__(decimals=decimals, **kwargs)
