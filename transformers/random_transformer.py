from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyRandomTransformer(CustomTransformer):
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return np.random.rand(*X.shape)
