"""Extract LIEF features from PE files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class PEExportsFeatures(CustomTransformer):
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
        return(bytez)
    
    
    def exports_features(self, lief_binary):
        from sklearn.feature_extraction import FeatureHasher

        exports = sorted(lief_binary.exported_functions)
        
        features_hashed = {}
        if exports:
            for i, x in enumerate(FeatureHasher(128, input_type='string').transform(exports).toarray()[0]):
                features_hashed.update({f'Exports_functions_hash_{i}': x})
        else:
            for i in range(128):
                features_hashed.update({f'Exports_functions_hash_{i}': 0})

        return features_hashed
    
    
    def get_exports_features(self, file_path):
        import lief
        try:
            pe_bytez = self.load_pe(file_path) 
            lief_binary = lief.PE.parse(list(pe_bytez))
            X = self.exports_features(lief_binary)
        
            return X

        except:
            X = {f'Exports_functions_hash_{i}': 0 for i in range(128)}
            return X
    

    def transform(self, X: dt.Frame):
        import pandas as pd

        ret_df = pd.DataFrame(
                [
                    self.get_exports_features(x)
                    for x in X.to_pandas().values[:,0]
                ]
            )
        
        self._output_feature_names = ret_df.columns.to_list()
        self._feature_desc = self._output_feature_names

        return ret_df
