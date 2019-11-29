"""Impute missing values by column mean"""
import datatable as dt
import numpy as np

from h2oaicore.systemutils import IgnoreError, dtype_global
from h2oaicore.transformer_utils import CustomTransformer


class MeanImputationTransformer(CustomTransformer):
    _mojo = True
    from h2oaicore.mojo import MojoWriter, MojoFrame

    def __init__(self, seed=12345, **kwargs):
        super().__init__(**kwargs)
        self.mean = None  # dt.Frame

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        if X[:, 0].countna1() < 1:
            # In case no missing values, this transformer makes no sense
            raise IgnoreError(self.__class__.__name__)
        X_pd = X.to_pandas()
        self.mean = X_pd.mean()
        X_pd = X_pd.fillna(self.mean)
        return X_pd

    def to_mojo(
        self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None
    ):
        import uuid
        from h2oaicore.mojo import MojoColumn, MojoFrame
        from h2oaicore.mojo_transformers import MjT_FillNa
        from h2oaicore.mojo_transformers_utils import AsType

        group_uuid = str(uuid.uuid4())
        group_name = self.__class__.__name__
        target_type = "float64"

        xnew = iframe[self.input_feature_names]
        oframe = MojoFrame()
        for col in xnew:
            ocol = MojoColumn(name=col.name, dtype=np.float64)
            out_frame = MojoFrame(columns=[ocol])
            in_frame = MojoFrame(columns=[col])
            in_frame = AsType(target_type).write_to_mojo(
                mojo, in_frame, group_uuid=group_uuid, group_name=group_name
            )
            mojo += MjT_FillNa(
                iframe=in_frame,
                oframe=out_frame,
                group_uuid=group_uuid,
                group_name=group_name,
                repl=self.mean[col.name].astype(target_type),
            )
            oframe += ocol
        oframe = AsType(target_type).write_to_mojo(
            mojo, oframe, group_uuid=group_uuid, group_name=group_name
        )
        return oframe
