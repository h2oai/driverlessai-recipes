"""Debiasing using reweighing"""

"""
This data recipe performs reweighing debiasing using the AIF360 package.  

https://github.com/Trusted-AI/AIF360
Kamiran, F., Calders, T. Data preprocessing techniques for classification without discrimination. 
Knowl Inf Syst 33, 1–33 (2012). https://doi.org/10.1007/s10115-011-0463-8

The transformer splits the original data as specified and returns training, validation, and test sets
with weights added.

1. Update the folder_path and data_file variables to indicate the location of the dataset(s).
2. validation_test_files lists additional validation or test files that need to be updated with weights.
3. validation_split indicates the percentiles at which the original data should be split to create a 
validation and test set.  If it's empty, no validation or test set is created.  [0.7] would create
a 70/30 training/validation split. [0.7, 0.9] would create a 70/20/10 training, validation, and test split.
4. target is the name of the target column.
5. favorable_label and unfavorable_label are the socially positive and negative target value respectively.
6. protected_group_info list of lists, where each sublist contains the name of a protected column,
the unprivledged level, and the privleged level.  Each of the protected columns must be binary.
7. From the Datasets section of driverless, click on ADD DATASET and then UPLOAD DATA RECIPE to upload this file.

Be sure to use the specified validation set to be used for validation when a model is trained.  The weights
can cause leakage if the validation or test data is used for determining the weights.
"""

import datatable as dt
import numpy as np
import os

from h2oaicore.data import CustomData
from h2oaicore.systemutils import config

_global_modules_needed_by_name = ['datetime', 'aif360', 'sklearn']


class MyData(CustomData):

    @staticmethod
    def create_data():

        _modules_needed_by_name = ['datetime', 'aif360', 'sklearn']

        import pandas as pd

        from aif360.datasets import BinaryLabelDataset
        from aif360.algorithms.preprocessing.reweighing import Reweighing

        """
        Update the below as needed
        """
        #########
        #########
        #########
        # Path to the data
        folder_path = 'tmp/'
        # Data file
        data_file = 'housing_train_proc.csv'

        validation_test_files = ['housing_test_proc.csv']

        validation_split = [0.6, 0.8]

        # Target column
        target = 'high_priced'
        favorable_label = 0
        unfavorable_label = 1

        # Privleged_group_info  = [[Protetected group name 1, prevleged level, unprivleged level], [Protetected group name 2, prevleged level, unprivleged level]]
        # The protected group columns need to be binary
        protected_group_info = [['hispanic', 0, 1], ['black', 0, 1]]
        #########
        #########
        #########

        # Set up protected group info
        protected_groups = [group_info[0] for group_info in protected_group_info]

        train = pd.read_csv(folder_path + data_file)
        dataset_orig = BinaryLabelDataset(df=train, label_names=[target], favorable_label=favorable_label,
                                          unfavorable_label=unfavorable_label,
                                          protected_attribute_names=protected_groups)

        privileged_groups = []
        unprivileged_groups = []
        for protected_group in protected_group_info:
            privileged_groups_dict = {}
            unprivileged_groups_dict = {}
            privileged_groups_dict[protected_group[0]] = protected_group[1]
            unprivileged_groups_dict[protected_group[0]] = protected_group[2]
            privileged_groups.append(privileged_groups_dict)
            unprivileged_groups.append(unprivileged_groups_dict)

        # Fit weights on the full dataset to be used on the external test set, if given
        RW_full = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        RW_full.fit(dataset_orig)

        # Split the original data into train, validation, and test if applicable
        if len(validation_split) == 1:
            dataset_orig_train, dataset_orig_valid = dataset_orig.split(validation_split, shuffle=True)
        elif len(validation_split) == 2:
            dataset_orig_train_valid, dataset_orig_test = dataset_orig.split([validation_split[1]], shuffle=True)
            # Fit the weights on both the validation and test set for the test set split
            RW_train_valid = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            RW_train_valid.fit(dataset_orig_train_valid)
            dataset_orig_train, dataset_orig_valid = dataset_orig_train_valid.split(
                [validation_split[0] / (validation_split[1])], shuffle=True)
        else:
            dataset_orig_train = dataset_orig

        # Fit weights on the training set only    
        RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        RW.fit(dataset_orig_train)
        dataset_transf_train = RW.transform(dataset_orig_train)

        # Add the weigts to the training set
        train_df = pd.DataFrame(dataset_transf_train.features, columns=dataset_transf_train.feature_names)
        train_df[target] = dataset_transf_train.labels.ravel()
        train_df['weights'] = dataset_transf_train.instance_weights.ravel()

        # Create datasets with minimum features calculated the given number of days ahead
        dataset_dict = {}
        dataset_dict[data_file.split('.')[0] + "_rw_train.csv"] = train_df

        # Add weights to the validation split (if a validation split was specified)
        if len(validation_split) >= 1:
            dataset_transf_valid = RW.transform(dataset_orig_valid)
            valid_df = pd.DataFrame(dataset_transf_valid.features, columns=dataset_transf_valid.feature_names)
            valid_df[target] = dataset_transf_valid.labels.ravel()
            valid_df['weights'] = dataset_transf_valid.instance_weights.ravel()
            dataset_dict[data_file.split('.')[0] + "_rw_validation.csv"] = valid_df

        # Add weights to the test split (if a test split was specified)
        if len(validation_split) >= 2:
            dataset_transf_test = RW_train_valid.transform(dataset_orig_test)
            test_df = pd.DataFrame(dataset_transf_test.features, columns=dataset_transf_test.feature_names)
            test_df[target] = dataset_transf_test.labels.ravel()
            test_df['weights'] = dataset_transf_test.instance_weights.ravel()
            dataset_dict[data_file.split('.')[0] + "_rw_test.csv"] = test_df

        # Add weights to the test files (If provided)       
        for valid_file in validation_test_files:
            valid = pd.read_csv(folder_path + valid_file)
            dataset_valid_orig = BinaryLabelDataset(df=valid, label_names=[target], favorable_label=favorable_label,
                                                    unfavorable_label=unfavorable_label,
                                                    protected_attribute_names=protected_groups)
            dataset_transf_valid = RW_full.transform(dataset_valid_orig)

            valid_df = pd.DataFrame(dataset_transf_valid.features, columns=dataset_transf_valid.feature_names)
            valid_df[target] = dataset_transf_valid.labels.ravel()
            valid_df['weights'] = dataset_transf_valid.instance_weights.ravel()

            dataset_dict[valid_file.split('.')[0] + "_rw_transformed.csv"] = valid_df

        return dataset_dict
