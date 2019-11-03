"""Extract LIEF features from PE files"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class PESectionCharacteristics(CustomTransformer):
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


    def section_features(self, lief_binary):
        from sklearn.feature_extraction import FeatureHasher
        
        sections = lief_binary.sections
        sections_list = [{'name': sec.name,
                          'size': sec.size,
                          'virtual_size': sec.virtual_size,
                          'entropy': sec.entropy,
                          'characteristics': [str(characteristic) for characteristic in sec.characteristics_lists]} for sec in sections]                

        features = [
            len(sections_list),
            sum([1 for sec in sections_list if sec['size'] == 0]),
            sum([1 for sec in sections_list if sec['name'] == ""])]

        section_characteristics = ['SECTION_CHARACTERISTICS.CNT_CODE',
                                   'SECTION_CHARACTERISTICS.CNT_INITIALIZED_DATA',
                                   'SECTION_CHARACTERISTICS.MEM_DISCARDABLE',
                                   'SECTION_CHARACTERISTICS.MEM_NOT_CACHED',
                                   'SECTION_CHARACTERISTICS.MEM_NOT_PAGED',
                                   'SECTION_CHARACTERISTICS.MEM_SHARED',
                                   'SECTION_CHARACTERISTICS.MEM_EXECUTE',
                                   'SECTION_CHARACTERISTICS.MEM_READ',
                                   'SECTION_CHARACTERISTICS.MEM_WRITE']
        for characteristic in section_characteristics:
            features.append(sum([1 for sec in sections_list if characteristic in sec['characteristics']]))

        sizes = {sec['name']: sec['size'] for sec in sections_list}
        features.extend(FeatureHasher(20).transform([sizes]).toarray()[0])
        
        virtual_sizes = {sec['name']: sec['virtual_size'] for sec in sections_list}
        features.extend(FeatureHasher(20).transform([virtual_sizes]).toarray()[0])
        
        entropies = {sec['name']: sec['entropy'] for sec in sections_list}
        features.extend(FeatureHasher(20).transform([entropies]).toarray()[0])
        
        characteristics = [characteristic for sec in sections_list for characteristic in sec['characteristics']]
        features.extend(FeatureHasher(20, input_type='string').transform([characteristics]).toarray()[0])
        
        return features

    
    def get_section_characteristics(self, file_path):
        import lief
        try:
            pe_bytez = self.load_pe(file_path) 
            lief_binary = lief.PE.parse(list(pe_bytez))
            X = self.section_features(lief_binary)
        
            return X

        except:
            X = np.zeros(92, dtype=np.float32)

            return X
    

    def transform(self, X: dt.Frame):
        
        import pandas as pd

        ret_df = pd.DataFrame(
                [
                    self.get_section_characteristics(x)
                    for x in X.to_pandas().values[:,0]
                ]
            )
        col_names = ['Sections_Count',
                     'Sections_Nonempty',
                     'Sections_No_Name']
        section_characteristics = ['SECTION_CHARACTERISTICS.CNT_CODE',
                                       'SECTION_CHARACTERISTICS.CNT_INITIALIZED_DATA',
                                       'SECTION_CHARACTERISTICS.MEM_DISCARDABLE',
                                       'SECTION_CHARACTERISTICS.MEM_NOT_CACHED',
                                       'SECTION_CHARACTERISTICS.MEM_NOT_PAGED',
                                       'SECTION_CHARACTERISTICS.MEM_SHARED',
                                       'SECTION_CHARACTERISTICS.MEM_EXECUTE',
                                       'SECTION_CHARACTERISTICS.MEM_READ',
                                       'SECTION_CHARACTERISTICS.MEM_WRITE']
        for characteristic in section_characteristics:
            col_names.append("Sections_{}".format(characteristic))
        col_names.extend(['Section_Sizes_Hash_{}'.format(x) for x in range(20)])
        col_names.extend(['Section_Virtual_Sizes_Hash_{}'.format(x) for x in range(20)])
        col_names.extend(['Section_Entropies_Hash_{}'.format(x) for x in range(20)])
        col_names.extend(['Section_Characteristics_Hash_{}'.format(x) for x in range(20)])
        self._output_feature_names = col_names
        self._feature_desc = col_names

        return ret_df
