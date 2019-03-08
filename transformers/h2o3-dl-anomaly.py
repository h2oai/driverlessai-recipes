from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config
import datatable as dt
import numpy as np
import h2o
import uuid
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


class MyH2OAutoEncoderAnomalyTransformer(CustomTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = str(uuid.uuid4())[:10]

    @staticmethod
    def get_default_properties():
        return dict(col_type="numcat", min_cols=1, max_cols=10, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        h2o.init()
        model = H2OAutoEncoderEstimator(activation='tanh', epochs=1, hidden=[50, 50])
        frame = h2o.H2OFrame(X.to_pandas())
        model.train(x=list(range(X.ncols)), training_frame=frame, model_id=self.id)
        return model.anomaly(frame).as_data_frame(header=False)

    def transform(self, X: dt.Frame):
        h2o.init()
        model = h2o.get_model(model_id=self.id)
        frame = h2o.H2OFrame(X.to_pandas())
        return model.anomaly(frame).as_data_frame(header=False)
