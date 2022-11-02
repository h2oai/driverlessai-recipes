"""Extract LIEF features from PE files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class PENormalizedByteCount(CustomTransformer):
    _unsupervised = True

    _regression = True
    _binary = True
    _multiclass = True
    _is_reproducible = True
    _parallel_task = True  # if enabled, params_base['n_jobs'] will be >= 1 (adaptive to system), otherwise 1
    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _can_use_multi_gpu = True  # if enabled, can get access to multiple GPUs for single transformer (experimental)
    _numeric_output = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def do_acceptance_test():
        return False

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def load_pe(self, file_path):
        with open(file_path, 'rb') as f:
            bytez = bytearray(f.read())
        return (bytez)

    def get_norm_byte_count(self, file_path):
        try:
            pe_bytez = self.load_pe(file_path)
            pe_int = np.frombuffer(pe_bytez, dtype=np.uint8)
            # Calculate normalized byte counts
            counts = np.bincount(pe_int, minlength=256)
            X = counts / counts.sum()

            return X

        except:
            X = np.zeros(256, dtype=np.float32)

            return X

    def transform(self, X: dt.Frame):

        import pandas as pd
        orig_col_name = X.names[0]
        ret_df = pd.DataFrame(
            [
                self.get_norm_byte_count(x)
                for x in X.to_pandas().values[:, 0]
            ]
        )
        self._output_feature_names = ['ByteNormCount_{}'.format(x) for x in range(ret_df.shape[1])]
        self._feature_desc = [f'Normalized Count of Byte value {x} for {orig_col_name} column' for x in
                              range(ret_df.shape[1])]
        return ret_df
