"""Rounds numbers to 1, 2 or 3 decimals"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyRoundTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

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

    _mojo = True
    from h2oaicore.mojo import MojoWriter, MojoFrame, MojoType

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        import uuid
        group_uuid = str(uuid.uuid4())
        group_name = "RoundTransformer"
        kws = dict()
        kws["op_name"] = "round"
        custom_param = dict()
        custom_param["decimals"] = (MojoType.INT32, self.decimals)
        kws["op_params"] = custom_param
        from h2oaicore.mojo import MojoColumn, MojoFrame
        from h2oaicore.mojo_transformers import MjT_CustomOp
        from h2oaicore.mojo_transformers_utils import AsType
        xnew = iframe[self.input_feature_names]
        oframe = MojoFrame()
        for col in xnew:
            ocol = MojoColumn(name=col.name, dtype=col.type)
            ocol_frame = MojoFrame(columns=[ocol])
            mojo += MjT_CustomOp(iframe=MojoFrame(columns=[col]), oframe=ocol_frame,
                                 group_uuid=group_uuid, group_name=group_name, kws)
            oframe += ocol
        oframe = AsType(dtype_global()).write_to_mojo(mojo, oframe,
                                                      group_uuid=group_uuid,
                                                      group_name=group_name)
        return oframe
