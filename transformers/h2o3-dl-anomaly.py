from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import os
import h2o
import uuid
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


class MyH2OAutoEncoderAnomalyTransformer(CustomTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = str(uuid.uuid4())[:10]
        self.raw_model_bytes = None

    @staticmethod
    def get_default_properties():
        return dict(col_type="numcat", min_cols=1, max_cols=10, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        h2o.init()
        model = H2OAutoEncoderEstimator(activation='tanh', epochs=1, hidden=[50, 50], reproducible=True, seed=1234)
        frame = h2o.H2OFrame(X.to_pandas())
        model_path = None
        try:
            model.train(x=list(range(X.ncols)), training_frame=frame, model_id=self.id)
            model_path = h2o.save_model(model=model)
            with open(model_path, "rb") as f:
                self.raw_model_bytes = f.read()
            return model.anomaly(frame).as_data_frame(header=False)
        finally:
            if model_path is not None:
                os.remove(model_path)
            h2o.remove(self.id)

    def transform(self, X: dt.Frame):
        h2o.init()
        with open(self.id, "wb") as f:
            f.write(self.raw_model_bytes)
        model = h2o.load_model(self.id)
        os.remove(self.id)
        frame = h2o.H2OFrame(X.to_pandas())
        return model.anomaly(frame).as_data_frame(header=False)
