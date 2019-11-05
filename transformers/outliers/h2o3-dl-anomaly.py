"""Anomaly score for each row based on reconstruction error of a H2O-3 deep learning autoencoder"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import os
import h2o
import uuid
from h2oaicore.systemutils import temporary_files_path, config
from h2o.estimators.deeplearning import H2OAutoEncoderEstimator


class MyH2OAutoEncoderAnomalyTransformer(CustomTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = None
        self.raw_model_bytes = None

    @staticmethod
    def get_default_properties():
        return dict(col_type="numcat", min_cols=2, max_cols=10, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        h2o.init(port=config.h2o_recipes_port)
        model = H2OAutoEncoderEstimator(activation='tanh', epochs=1, hidden=[50, 50], reproducible=True, seed=1234)
        frame = h2o.H2OFrame(X.to_pandas())
        model_path = None
        try:
            model.train(x=list(range(X.ncols)), training_frame=frame)
            self.id = model.model_id
            model_path = os.path.join(temporary_files_path, "h2o_model." + str(uuid.uuid4()))
            model_path = h2o.save_model(model=model, path=model_path)
            with open(model_path, "rb") as f:
                self.raw_model_bytes = f.read()
            return model.anomaly(frame).as_data_frame(header=False)
        finally:
            if model_path is not None:
                os.remove(model_path)
            h2o.remove(model)

    def transform(self, X: dt.Frame):
        h2o.init(port=config.h2o_recipes_port)
        model_path = os.path.join(temporary_files_path, self.id)
        with open(model_path, "wb") as f:
            f.write(self.raw_model_bytes)
        model = h2o.load_model(model_path)
        os.remove(model_path)
        frame = h2o.H2OFrame(X.to_pandas())
        try:
            return model.anomaly(frame).as_data_frame(header=False)
        finally:
            h2o.remove(self.id)
