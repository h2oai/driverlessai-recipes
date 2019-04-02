from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyTwoSigmaOutlierRemoverTransformer(CustomTransformer):
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.mean = X.mean1()
        self.sd = X.sd1()
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = dt.Frame(X)
        X[self.mean - 2 * self.sd > dt.f[0], :] = self.mean - 2 * self.sd
        X[self.mean + 2 * self.sd < dt.f[0], :] = self.mean + 2 * self.sd
        return X
