"""Extract LIEF features from PE files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class PEGeneralFeatures(CustomTransformer):
    _modules_needed_by_name = ['lief==0.9.0']
    _regression = True
    _binary = True
    _multiclass = True
    _is_reproducible = True
    _parallel_task = True  # if enabled, params_base['n_jobs'] will be >= 1 (adaptive to system), otherwise 1
    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _can_use_multi_gpu = True  # if enabled, can get access to multiple GPUs for single transformer (experimental)
    _numeric_output = True

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

    def get_general_features(self, file_path):
        import lief
        try:
            pe_bytez = self.load_pe(file_path)
            lief_binary = lief.PE.parse(list(pe_bytez))
            X = {'exports_count': len(lief_binary.exported_functions),
                 'imports_count': len(lief_binary.imported_functions),
                 'has_configuration': int(lief_binary.has_configuration),
                 'has_debug': int(lief_binary.has_debug),
                 'has_exceptions': int(lief_binary.has_exceptions),
                 'has_nx': int(lief_binary.has_nx),
                 'has_relocations': int(lief_binary.has_relocations),
                 'has_resources': int(lief_binary.has_resources),
                 'has_rich_header': int(lief_binary.has_rich_header),
                 'has_signature': int(lief_binary.has_signature),
                 'has_tls': int(lief_binary.has_tls),
                 'libraries_count': len(lief_binary.libraries),
                 'size': len(pe_bytez),
                 'symbols_count': len(lief_binary.symbols),
                 'virtual_size': lief_binary.virtual_size}

            return X

        except:
            X = {'exports_count': 0,
                 'imports_count': 0,
                 'has_configuration': 0,
                 'has_debug': 0,
                 'has_exceptions': 0,
                 'has_nx': 0,
                 'has_relocations': 0,
                 'has_resources': 0,
                 'has_rich_header': 0,
                 'has_signature': 0,
                 'has_tls': 0,
                 'libraries_count': 0,
                 'size': 0,
                 'symbols_count': 0,
                 'virtual_size': 0}

            return X

    def transform(self, X: dt.Frame):

        import pandas as pd

        ret_df = pd.DataFrame(
            [
                self.get_general_features(x)
                for x in X.to_pandas().values[:, 0]
            ]
        )

        self._output_feature_names = ['General_{}'.format(x) for x in ret_df.columns.to_list()]
        self._feature_desc = self._output_feature_names

        return ret_df
