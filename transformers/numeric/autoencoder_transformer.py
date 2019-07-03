"""
Autoencoder work in progress.
"""

import datatable as dt
import numpy as np
import os
import uuid

from h2oaicore.models import TensorFlowModel
from h2oaicore.systemutils import config, release_gpus, temporary_files_path, DummyContext
from h2oaicore.transformer_utils import CustomTransformer

from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning

class AutoencoderTransformer(CustomTransformer, TensorFlowModel):

    _tensorflow = True
    _parallel_task = True  # assumes will use n_jobs in params_base
    _can_use_gpu = True
    _can_use_multi_gpu = True
    _is_reproducible = False
    _can_handle_non_numeric = False
    _regression = True
    _binary = True
    _multiclass = True


    def __init__(self, tmp_dir=temporary_files_path, **kwargs):
        super().__init__(**kwargs)
        from sklearn.preprocessing import MinMaxScaler
        self.sc = MinMaxScaler((0, 1))

    #@staticmethod
    #def is_enabled():
    #    return True

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols="all", max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        # don't use GPU memory unless actually found relevant data
        import h2oaicore.keras as keras
        self.tf_config = self.ConfigProto()
        self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        keras.backend.set_session(session=TensorFlowModel.make_sess(self.tf_config))

        from h2oaicore.keras import layers
        from h2oaicore.keras import Model
        from h2oaicore.keras import optimizers

        X = X.to_numpy()
        train_features = self.sc.fit_transform(X)

        inp = layers.Input(shape=(train_features.shape[1],))

        n = max(1, train_features.shape[1] // 2)

        x = layers.Dense(n, use_bias=False)(inp)
        x = layers.BatchNormalization(scale=False)(x)
        h1_encoded = layers.Activation('tanh')(x)

        x = layers.Dense(train_features.shape[1], use_bias=True)(h1_encoded)
        decoded = layers.Activation('sigmoid')(x)

        autoencoder = Model(inp, decoded)
        h1_encoder = Model(inp, h1_encoded)

        autoencoder.compile(optimizer=optimizers.SGD(0.01, momentum=0.95, nesterov=False), loss='mse')

        #mcp = keras.callbacks.ModelCheckpoint(self.dl_text_model_path, verbose=1, save_best_only=True, save_weights_only=True)

        autoencoder.fit(
                train_features, 
                train_features,
                epochs=1,
                #steps_per_epoch=1000,
                batch_size=64,
                shuffle=True,
                )

        self.model = h1_encoder
        self.model_weights = h1_encoder.get_weights()

        return self.transform(dt.Frame(X))

    def transform(self, X: dt.Frame):
        import h2oaicore.keras as keras
        X = X.to_numpy()
        features = self.sc.transform(X)
        self.model.set_weights(self.model_weights)
        return dt.Frame(self.model.predict(features))

