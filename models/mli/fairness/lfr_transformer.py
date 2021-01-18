"""AIF360 LFR"""

"""
AIF 360 LFR
"""

import datatable as dt
import numpy as np
import os

from h2oaicore.data import CustomData
from h2oaicore.systemutils import config

_global_modules_needed_by_name = ['aif360', 'sklearn', 'scipy'] 

class MyData(CustomData):
    
    @staticmethod
    def create_data():
        
        _modules_needed_by_name = [ 'aif360', 'sklearn', 'scipy']        

        import pandas as pd
        import numba
        #from numba.decorators import jit

        from aif360.datasets import BinaryLabelDataset
        #from aif360.datasets import AdultDataset
        #from aif360.metrics import BinaryLabelDatasetMetric
        #from aif360.metrics import ClassificationMetric
        #from aif360.metrics.utils import compute_boolean_conditioning_vector
        from aif360.algorithms.preprocessing.lfr_helpers import helpers as lfr_helpers
        from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
        from aif360.algorithms.preprocessing.lfr import LFR
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        
        
        # Test and add option of a dataset split
        """
        Update the below as needed
        """
        # Path to the data
        folder_path = 'tmp/'  
        # Data file
        #data_file = 'housing_train_proc.csv' # Data file
        data_file = './tmp/kim.montgomery@h2o.ai/7ceffc52-5890-11eb-b9ca-023b14eb1bf5/housing_train_proc.csv.1610866316.2284853.bin'
        
        #validation_test_files = ['housing_test_proc.csv']
        validation_test_files = []
        
        validation_split = [0.6, 0.8]
        
        # Add parameter scanning once the basic model works
        k=10
        Ax=0.1
        Ay=1.0
        Az=2.0
        maxiter=5000
        maxfun=5000

        # Target column
        target = 'high_priced'
        # The protected group columns need to be binary
        # Only one privleged group is supported for this recipe
        protected_group_info = [['hispanic', 1, 0]]

        
        protected_groups = [group_info[0] for group_info in protected_group_info]
       
        #train_file = 'housing_train_proc.csv'
        train = pd.read_csv(folder_path + data_file)
        dataset_orig = BinaryLabelDataset(df=train, label_names=[target], protected_attribute_names=protected_groups)
        dataset_orig_full = BinaryLabelDataset(df=train, label_names=[target], protected_attribute_names=protected_groups)        
        
        # Set up protected group info
        privileged_groups = []
        unprivileged_groups = []
        for protected_group in protected_group_info:
            privileged_groups_dict = {}
            unprivileged_groups_dict = {}
            privileged_groups_dict[protected_group[0]] = protected_group[1]
            unprivileged_groups_dict[protected_group[0]] = protected_group[2]
            privileged_groups.append(privileged_groups_dict)
            unprivileged_groups.append(unprivileged_groups_dict)
        
        
        # Scale data
        scale_orig_full = StandardScaler()
        dataset_orig_full.features = scale_orig_full.fit_transform(dataset_orig_full.features)
          
        TR_full = LFR(unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups,
                      k=k, Ax=Ax, Ay=Ay, Az=Az, 
                      verbose=1)
        TR_full.fit(dataset_orig_full, maxiter=maxiter, maxfun=maxfun)       
        
        
        if len(validation_split) == 1:
            dataset_orig_train, dataset_orig_valid = dataset_orig.split(validation_split, shuffle=True)
        elif len(validation_split) == 2:
            dataset_orig_train_valid, dataset_orig_test = dataset_orig.split([validation_split[1]], shuffle=True)
            # Fit the weights on both the validation and test set for the test set split
            
            scale_orig_train_valid  = StandardScaler()
            dataset_orig_train_valid.features = scale_orig_train_valid.fit_transform(dataset_orig_train_valid.features)
          
            TR_train_valid  = LFR(unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups,
                                  k=k, Ax=Ax, Ay=Ay, Az=Az, 
                                  verbose=1)
            TR_train_valid.fit(dataset_orig_train, maxiter=maxiter, maxfun=maxfun)    
            
            dataset_orig_train, dataset_orig_valid = dataset_orig_train_valid.split([validation_split[0]/(validation_split[1])], shuffle=True)
        else:
            dataset_orig_train = dataset_orig
            
        # Fit weights on the training set only    
        
        scale_orig = StandardScaler()
        dataset_orig_train.features = scale_orig.fit_transform(dataset_orig_train.features)
          
        TR = LFR(unprivileged_groups=unprivileged_groups,
                 privileged_groups=privileged_groups,
                 k=k, Ax=Ax, Ay=Ay, Az=Az, 
                 verbose=1)
        TR.fit(dataset_orig_train, maxiter=maxiter, maxfun=maxfun)            
        
        dataset_transf_train = TR.transform(dataset_orig_train)
        
        # Add the weigts to the training set
        train_df = pd.DataFrame(dataset_transf_train.features, columns=dataset_transf_train.feature_names)
        train_df[target] = dataset_transf_train.labels.ravel()
        
        # Create datasets with minimum features calculated the given number of days ahead
        dataset_dict = {}
        dataset_dict[data_file.split('.')[0] + "_rw_train.csv"] = train_df
        
        # Add weights to the validation split (if a validation split was specified)
        if len(validation_split) >= 1:
            #dataset_transf_valid = RW.transform(dataset_orig_valid)
            dataset_orig_valid.features = scale_orig.transform(dataset_orig_valid.features)        
            dataset_transf_valid = TR.transform(dataset_orig_valid)
            valid_df = pd.DataFrame(dataset_transf_valid.features, columns=dataset_transf_valid.feature_names)
            valid_df[target] = dataset_transf_valid.labels.ravel()
            #valid_df['weights'] = dataset_transf_valid.instance_weights.ravel()
            dataset_dict[data_file.split('.')[0] + "_rw_validation.csv"] = valid_df
                 
        # Add weights to the test split (if a test split was specified)
        if len(validation_split) >= 2:
            #dataset_transf_test = RW_train_valid.transform(dataset_orig_test)
            #
            dataset_orig_test.features = scale_orig_train_valid.transform(dataset_orig_test.features)        
            dataset_transf_test = TR.transform(dataset_orig_test)
            test_df = pd.DataFrame(dataset_transf_test.features, columns=dataset_transf_test.feature_names)
            test_df[target] = dataset_transf_test.labels.ravel()
            #test_df['weights'] = dataset_transf_test.instance_weights.ravel()  
            dataset_dict[data_file.split('.')[0] + "_rw_test.csv"] = test_df
                
        # Add weights to the test files (If provided)       
        for valid_file in  validation_test_files:
            valid = pd.read_csv(folder_path + valid_file)
            dataset_valid_orig = BinaryLabelDataset(df=valid, label_names=[target], protected_attribute_names=protected_groups)
            dataset_valid_orig.features = scale_orig_full.transform(dataset_valid_orig.features)        
            dataset_transf_valid = TR_full.transform(dataset_valid_orig)
                    
            valid_df = pd.DataFrame(dataset_transf_valid.features, columns=dataset_transf_valid.feature_names)
            valid_df[target] = dataset_transf_valid.labels.ravel()
            #valid_df['weights'] = dataset_transf_valid.instance_weights.ravel()
            
            dataset_dict[valid_file.split('.')[0] + "_rw_transformed.csv"] = valid_df

            
        return dataset_dict