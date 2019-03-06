from h2oaicore.systemutils import segfault
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyLogTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=3, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, [dt.log(dt.f[i]) for i in range(X.ncols)]]

    # optional
    _mojo = True
    from h2oaicore.mojo import MojoWriter, MojoFrame

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame):
        from h2oaicore.mojo import MojoColumn, MojoFrame
        from h2oaicore.mojo_transformers import MjT_Log
        xnew = iframe[self.input_feature_names]
        oframe = MojoFrame()
        for col in xnew:
            ocol = MojoColumn(name=col.name, dtype=np.float64)
            ocol_frame = MojoFrame(columns=[ocol])
            mojo += MjT_Log(iframe=MojoFrame(columns=[col]), oframe=ocol_frame)
            oframe += ocol
        return oframe


class MyExpDiffTransformer(CustomTransformer):
    _interpretability = 10
    _interpretability_min = 3

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=2, max_cols=2, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, dt.exp(dt.f[0] - dt.f[1])]


class MyStrLenEncoderTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].str.len()


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


class MySegfaultTransformer(CustomTransformer):
    _is_enabled = False

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        segfault()

    def transform(self, X: dt.Frame):
        segfault()


class MyRandomTransformer(CustomTransformer):
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return np.random.rand(X.shape)
