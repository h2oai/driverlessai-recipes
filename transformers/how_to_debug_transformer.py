import datatable as dt
import pandas as pd
import numpy as np


# from h2oaicore.transformer_utils import CustomTransformer

def debug_data():
    X = dt.fread("../data/weather_missing.csv")
    target = 'RainTomorrow'
    y = X[:, target]
    return X, y, target


# Emulate CustomTransformer
class CustomTransformer:

    def __init__(self):
        X, y, target = debug_data()
        self.feature_names = [x for x in X.names if x != target]
        self.num_classes = y.nunique1()
        print(self.feature_names)
        print(self.num_classes)


class MyTransformer(CustomTransformer):
    @staticmethod
    def is_enabled():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="numcat", min_cols="all", max_cols="all", relative_importance=1)

    # Train
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        X_num = X[:, [float, int]]
        X_cat = X[:, [int]]
        if X_num.ncols == 0 or X_cat.ncols == 0:
            return np.zeros(X.shape)

        self.means = {}
        for num in X_num.names:
            # X[:, num+"_sorted"] = X[:, num].sort(0)
            # print(X[:, num+"_sorted"].to_pandas())
            for cat in X_cat.names:
                key = (num, cat)
                self.means[key] = X[:, dt.mean(dt.f[num]), dt.by(cat)][:, -1]
                # print("key %s" % str(key))
                # print(self.means[key].to_pandas())
        return X_num[:, 0]

    # Validate
    def transform(self, X: dt.Frame):
        X_num = X[:, [float, int]]
        return X_num[:, 0]


# name method test_xxx to run it with `pytest -s how_to_debug_transformer.py` or `python how_to_debug_transformer.py`
def test_transformer():
    X, y, target = debug_data()
    tr = MyTransformer()

    X_munged = tr.fit_transform(X, y)
    assert X_munged.shape[0] == X.nrows

    X_munged2 = tr.transform(X)
    assert X_munged2.shape[0] == X.nrows


if __name__ == '__main__':
    test_transformer()
