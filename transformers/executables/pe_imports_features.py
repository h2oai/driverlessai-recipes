"""Extract LIEF features from PE files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class PEImportsFeatures(CustomTransformer):
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
    
    
    def imports_features(self, lief_binary):
        from sklearn.feature_extraction import FeatureHasher

        imports = lief_binary.imports
        features = {}
        for lib in imports:
            if lib.name not in features:
                features[lib.name] = []
            for entry in lib.entries:
                if entry.is_ordinal:
                    features[lib.name].append("ordinal" + str(entry.ordinal))
                else:
                    features[lib.name].append(entry.name[:10000])
        
        features_hashed = {}
        libraries = sorted(list(set([l.lower() for l in features.keys()])))
        for i, x in enumerate(FeatureHasher(256, input_type='string').transform([libraries]).toarray()[0]):
            features_hashed.update({f'Imports_libraries_hash_{i}': x})
        entries = sorted([lib.lower() + ':' + e for lib, elist in features.items() for e in elist])
        for i, x in enumerate(FeatureHasher(1024, input_type='string').transform([entries]).toarray()[0]):
            features_hashed.update({f'Imports_entries_hash_{i}': x})   
        return features_hashed
    
    
    def get_imports_features(self, file_path):
        import lief
        try:
            pe_bytez = self.load_pe(file_path) 
            lief_binary = lief.PE.parse(list(pe_bytez))
            X = self.imports_features(lief_binary)
        
            return X

        except:
            X = {f'Imports_libraries_hash_{i}': 0 for i in range(256)}
            X.update({f'Imports_entries_hash_{i}':0 for i in range(1024)})
            return X
    

    def transform(self, X: dt.Frame):
        import pandas as pd

        ret_df = pd.DataFrame(
                [
                    self.get_imports_features(x)
                    for x in X.to_pandas().values[:,0]
                ]
            )
        
        self._output_feature_names = ret_df.columns.to_list()
        self._feature_desc = self._output_feature_names

        return ret_df
