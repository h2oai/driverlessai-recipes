"""Converts numbers to the square root, preserving the sign of the original numbers"""
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
        return X[:, [(dt.f[i]/dt.abs(dt.f[i]))*dt.exp(0.5*dt.log(dt.abs(dt.f[i]))) for i in range(X.ncols)]]

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