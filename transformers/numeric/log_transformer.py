"""Converts numbers to their Logarithm"""
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import dtype_global
import datatable as dt
import numpy as np


class MyLogTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=3, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        if dtype_global() == np.float32:
            return X[:, [dt.float32(dt.log(dt.f[i])) for i in range(X.ncols)]]
        else:
            return X[:, [dt.float64(dt.log(dt.f[i])) for i in range(X.ncols)]]

    # optional
    _mojo = True
    from h2oaicore.mojo import MojoWriter, MojoFrame

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        import uuid
        group_uuid = str(uuid.uuid4())
        group_name = self.__class__.__name__
        from h2oaicore.mojo import MojoColumn, MojoFrame
        from h2oaicore.mojo_transformers import MjT_Log
        from h2oaicore.mojo_transformers_utils import AsType
        xnew = iframe[self.input_feature_names]
        oframe = MojoFrame()
        for col in xnew:
            ocol = MojoColumn(name=col.name, dtype=np.float64)
            ocol_frame = MojoFrame(columns=[ocol])
            mojo += MjT_Log(iframe=MojoFrame(columns=[col]), oframe=ocol_frame,
                            group_uuid=group_uuid, group_name=group_name)
            oframe += ocol
        oframe = AsType(dtype_global()).write_to_mojo(mojo, oframe,
                                                      group_uuid=group_uuid,
                                                      group_name=group_name)
        return oframe
