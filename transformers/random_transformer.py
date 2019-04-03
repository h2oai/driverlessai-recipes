from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyRandomTransformer(CustomTransformer):
    def __init__(self, seed=12345, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        np.random.seed(self.seed)
        return np.random.rand(*X.shape)
