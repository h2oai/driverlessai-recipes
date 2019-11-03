"""Extract LIEF features from PE files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class PEDataDirectoryFeatures(CustomTransformer):
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
    
    
    def data_directory_features(self, lief_binary):
        
        data_directories = lief_binary.data_directories
        features = {}
        for data_directory in data_directories:
            features.update({'Data_Directory_{}_size'.format(str(data_directory.type).split(".")[1]): data_directory.size})
            features.update({'Data_Directory_{}_virtual_address'.format(str(data_directory.type).split(".")[1]): data_directory.rva})
        return features
    
    
    def get_data_directory_features(self, file_path):
        import lief
        try:
            pe_bytez = self.load_pe(file_path) 
            lief_binary = lief.PE.parse(list(pe_bytez))
            X = self.data_directory_features(lief_binary)
        
            return X

        except:
            X = {'Data_Directory_EXPORT_TABLE_size': 0,
                 'Data_Directory_EXPORT_TABLE_virtual_address': 0,
                 'Data_Directory_IMPORT_TABLE_size': 0,
                 'Data_Directory_IMPORT_TABLE_virtual_address': 0,
                 'Data_Directory_RESOURCE_TABLE_size': 0,
                 'Data_Directory_RESOURCE_TABLE_virtual_address': 0,
                 'Data_Directory_EXCEPTION_TABLE_size': 0,
                 'Data_Directory_EXCEPTION_TABLE_virtual_address': 0,
                 'Data_Directory_CERTIFICATE_TABLE_size': 0,
                 'Data_Directory_CERTIFICATE_TABLE_virtual_address': 0,
                 'Data_Directory_BASE_RELOCATION_TABLE_size': 0,
                 'Data_Directory_BASE_RELOCATION_TABLE_virtual_address': 0,
                 'Data_Directory_DEBUG_size': 0,
                 'Data_Directory_DEBUG_virtual_address': 0,
                 'Data_Directory_ARCHITECTURE_size': 0,
                 'Data_Directory_ARCHITECTURE_virtual_address': 0,
                 'Data_Directory_GLOBAL_PTR_size': 0,
                 'Data_Directory_GLOBAL_PTR_virtual_address': 0,
                 'Data_Directory_TLS_TABLE_size': 0,
                 'Data_Directory_TLS_TABLE_virtual_address': 0,
                 'Data_Directory_LOAD_CONFIG_TABLE_size': 0,
                 'Data_Directory_LOAD_CONFIG_TABLE_virtual_address': 0,
                 'Data_Directory_BOUND_IMPORT_size': 0,
                 'Data_Directory_BOUND_IMPORT_virtual_address': 0,
                 'Data_Directory_IAT_size': 0,
                 'Data_Directory_IAT_virtual_address': 0,
                 'Data_Directory_DELAY_IMPORT_DESCRIPTOR_size': 0,
                 'Data_Directory_DELAY_IMPORT_DESCRIPTOR_virtual_address': 0,
                 'Data_Directory_CLR_RUNTIME_HEADER_size': 0,
                 'Data_Directory_CLR_RUNTIME_HEADER_virtual_address': 0}
            return X
    

    def transform(self, X: dt.Frame):
        import pandas as pd

        ret_df = pd.DataFrame(
                [
                    self.get_data_directory_features(x)
                    for x in X.to_pandas().values[:,0]
                ]
            )
        
        self._output_feature_names = ret_df.columns.to_list()
        self._feature_desc = self._output_feature_names

        return ret_df
