"""Extract LIEF features from PE files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class PEHeaderFeatures(CustomTransformer):
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
    
    
    def header_features(self, lief_binary):
        from sklearn.feature_extraction import FeatureHasher
        
        header = lief_binary.header
        opt_header = lief_binary.optional_header

        features = {}
        # Header features
        features['Header_time_date_stamps'] = header.time_date_stamps
        features['Header_sizeof_optional_header'] = header.sizeof_optional_header
        for i, x in enumerate(FeatureHasher(10, input_type='string').transform([str(header.machine)]).toarray()[0]):
            features.update({f'Header_machine_hash_{i}': x})
        for i, x in enumerate(FeatureHasher(10, input_type='string').transform([str(c) for c in header.characteristics_list]).toarray()[0]):
            features.update({f'Header_characteristics_hash_{i}': x}) 
        # Optional Header features
        for i, x in enumerate(FeatureHasher(10, input_type='string').transform([str(opt_header.subsystem)]).toarray()[0]):
            features.update({f'Optional_Header_subsystem_hash_{i}': x})
        for i, x in enumerate(FeatureHasher(10, input_type='string').transform([str(c) for c in opt_header.dll_characteristics_lists]).toarray()[0]):
            features.update({f'Optional_Header_dll_characteristics_hash_{i}': x}) 
        for i, x in enumerate(FeatureHasher(10, input_type='string').transform([str(opt_header.magic)]).toarray()[0]):
            features.update({f'Optional_Header_magic_hash_{i}': x})
        features['major_image_version'] = opt_header.major_image_version
        features['minor_image_version'] = opt_header.minor_image_version
        features['major_linker_version'] = opt_header.major_linker_version
        features['minor_linker_version'] = opt_header.minor_linker_version
        features[
            'major_operating_system_version'] = opt_header.major_operating_system_version
        features[
            'minor_operating_system_version'] = opt_header.minor_operating_system_version
        features['major_subsystem_version'] = opt_header.major_subsystem_version
        features['minor_subsystem_version'] = opt_header.minor_subsystem_version
        features['sizeof_code'] = opt_header.sizeof_code
        features['sizeof_headers'] = opt_header.sizeof_headers
        features['sizeof_heap_commit'] = opt_header.sizeof_heap_commit
        return features
    
    
    def get_header_features(self, file_path):
        import lief
        try:
            pe_bytez = self.load_pe(file_path) 
            lief_binary = lief.PE.parse(list(pe_bytez))
            X = self.header_features(lief_binary)
        
            return X

        except:
            X = {'Header_time_date_stamps': 0,
                 'Header_sizeof_optional_header': 0}
            X.update({f'Header_machine_hash_{i}': 0 for i in range(10)})
            X.update({f'Header_characteristics_hash_{i}':0 for i in range(10)})
            X.update({f'Optional_Header_subsystem_hash_{i}':0 for i in range(10)})
            X.update({f'Optional_Header_dll_characteristics_hash_{i}':0 for i in range(10)})
            X.update({f'Optional_Header_magic_hash_{i}':0 for i in range(10)})
            X.update({f'{feature_name}': 0 for feature_name in ['major_image_version', 
                                                                 'minor_image_version', 
                                                                 'major_linker_version', 
                                                                 'minor_linker_version', 
                                                                 'major_operating_system_version', 
                                                                 'minor_operating_system_version', 
                                                                 'major_subsystem_version', 
                                                                 'minor_subsystem_version', 
                                                                 'sizeof_code', 
                                                                 'sizeof_headers', 
                                                                 'sizeof_heap_commit']})
            return X
    

    def transform(self, X: dt.Frame):
        
        import pandas as pd

        ret_df = pd.DataFrame(
                [
                    self.get_header_features(x)
                    for x in X.to_pandas().values[:,0]
                ]
            )
        
        self._output_feature_names = ret_df.columns.to_list()
        self._feature_desc = self._output_feature_names

        return ret_df
