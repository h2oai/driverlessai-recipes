"""Rounds numbers to 1, 2 or 3 decimals"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class MyRoundTransformer(CustomTransformer):
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def get_parameter_choices():
        return {"decimals": [2,3,4]}

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

    mojo = True
    from h2oaicore.mojo import MojoWriter, MojoFrame

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        from model2proto import Pipeline_pb2
        import os
        import tempfile
        import zipfile

        mojo_pipeline = Pipeline_pb2.Pipeline()
        mojo_pipeline.mojo_version = 2
        mojo_pipeline.uuid = "c30815f6-f6cb-475d-9f32-64d4152bce2d"

        ziph = zipfile.ZipFile("transform_custome_round.mojo", mode="w",
                               compression=zipfile.ZIP_DEFLATED)

        feature_frame = mojo_pipeline.features
        col_A = feature_frame.columns.add()
        col_A.name = "A"
        col_A.float64_type.SetInParent()

        output_frame = mojo_pipeline.outputs
        col_B = output_frame.columns.add()
        col_B.name = "B"
        col_B.float64_type.SetInParent()

        col_C = output_frame.columns.add()
        col_C.name = "C"
        col_C.float64_type.SetInParent()

        col_D = output_frame.columns.add()
        col_D.name = "D"
        col_D.float64_type.SetInParent()

        round_decimals = Pipeline_pb2.Int32Array()
        round_decimals.values.append(1)
        round_decimals.values.append(2)
        round_decimals.values.append(3)

        tmp_fname = tempfile.NamedTemporaryFile().name

        with open(tmp_fname, "wb") as f:
            f.write(round_decimals.SerializeToString())

        ziph.write(tmp_fname, arcname=os.path.join('mojo', 'round_decimals'),
                   compress_type=zipfile.ZIP_STORED)

        custom_tran = mojo_pipeline.transformations.add()
        custom_tran.inputs.append("A")
        custom_tran.outputs.append("B")
        custom_tran.outputs.append("C")
        custom_tran.outputs.append("D")
        custom_op = custom_tran.custom_op
        custom_op.transformer_name = "RoundTransform"

        round_param = Pipeline_pb2.CustomParam()
        round_param.name = "decimals"
        decimal_arr = round_param.binary_param
        decimal_arr.data_type = Pipeline_pb2.INT32
        decimal_arr.file_name = os.path.join('mojo', 'round_decimals')
        custom_op.params.append(round_param)

        tmp_fname = tempfile.NamedTemporaryFile().name

        with open(tmp_fname, "wb") as f:
            f.write(mojo_pipeline.SerializeToString())

        ziph.write(tmp_fname, arcname=os.path.join('mojo', 'pipeline.pb'),
                   compress_type=zipfile.ZIP_STORED)

        ziph.close()
        
        import uuid
        group_uuid = str(uuid.uuid4())
        group_name = self.__class__.__name__
        from h2oaicore.mojo import MojoColumn, MojoFrame
        from h2oaicore.mojo_transformers import MjT_CustomOp
        from h2oaicore.mojo_transformers_utils import AsType
        xnew = iframe[self.input_feature_names]
        oframe = MojoFrame()
        for col in xnew:
            ocolB = MojoColumn(name=col.name, dtype=np.float64)
            ocolC = MojoColumn(name=col.name, dtype=np.float64)
            ocolD = MojoColumn(name=col.name, dtype=np.float64)
            ocol_frame = MojoFrame(columns=[ocolB, ocolC, ocolD])
            mojo += MjT_CustomOp(iframe=MojoFrame(columns=[col]), oframe=ocol_frame,
                            group_uuid=group_uuid, group_name=group_name)
            oframe += ocolB
            oframe += ocolC
            oframe += ocolD
        oframe = AsType(dtype_global()).write_to_mojo(mojo, oframe,
                                                      group_uuid=group_uuid,
                                                      group_name=group_name)
        return oframe
