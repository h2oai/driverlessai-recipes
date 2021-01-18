"""Adverserial debiasing """

import uuid
import os
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count
from h2oaicore.systemutils import user_dir, remove, config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdebug


import uuid
import os
import datatable as dt
import numpy as np
from h2oaicore.models import CustomTensorFlowModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count, loggerdata, load_obj_bytes
from h2oaicore.systemutils import user_dir, remove, config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdebug
import functools

import numpy as np
import pandas as pd
import tensorflow as tf


class Adversarial_Debiasing42(CustomTensorFlowModel):
    """
        TensorFlow-based Custom Model
    """
    _tensorflow = True
    _parallel_task = True
    _can_use_gpu = True
    _can_use_multi_gpu = True

    _regression = False
    _binary = True
    _multiclass = False
    _display_name = "AD42"
    _description = "AD42"
    _modules_needed_by_name = ['shap', 'scipy', 'pandas', 'matplotlib', 'AIF360', 'sklearn']

    _is_reproducible = False
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _mojo = False


    @staticmethod
    def do_acceptance_test():
        return False
    
    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):

        self.params = dict(random_state=kwargs.get("random_state", 24),
                           classifier_num_hidden_units = 20,
                           num_epochs = 100, batch_size = 128,
                           adversary_loss_weight=0.1)

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            classifier_num_hidden_units = [50, 100, 200, 1000]
            batch_size = [32, 128, 256, 512]
            num_epochs = [100, 1000]
            adversary_loss_weight = [0.05*ii for ii in range(1, 14)]

        elif accuracy >= 5:
            classifier_num_hidden_units = [50, 100, 200]
            batch_size = [32, 128, 256, 512]
            num_epochs = [100, 1000]
            adversary_loss_weight = [0.05*ii for ii in range(1, 14)]
            
        else:
            classifier_num_hidden_units = [50, 200]
            batch_size = [32, 128, 512]
            num_epochs = [100]
            adversary_loss_weight = [0.05*ii for ii in range(1, 14)]


        self.params["classifier_num_hidden_units"] = np.random.choice(classifier_num_hidden_units)
        self.params["batch_size"] = np.random.choice(batch_size)
        self.params["num_epochs"] = np.random.choice(num_epochs)        
        self.params["adversary_loss_weight"] = np.random.choice(adversary_loss_weight)    


    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files 
        # Set the default value without context available (required to pass acceptance test)
        tmp_folder = os.path.join(user_dir(), "%s_AD_model_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_AD_model_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "AD was denied temp folder creation rights")
            tmp_folder = os.path.join(user_dir(), "%s_AD_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "AD temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_AD_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(user_dir(), "%s_AD_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "AD temp folder {}".format(tmp_folder))
        return tmp_folder


    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
             
        # Specify these parameters for the dataset.
        #
        # Also set feature engineering effort to 0
        # under the features section of expert settings.
        ########################
        # Specify the protected column.
        # The protected column must be numeric.
        #self.protected_name = "black"
        # Specify the level of the protected group in the protected column
        #self.protected_label = 1
        # Specify the target level considered to be a positive outcome
        # Must be encoded as 0/1
        #self.positive_target = 0
        # Set minimum mean protected ratio needed to avoid a penalty 
        # (mean protected ratio = mean predictions for the protected group/mean predictions for all other groups)
        #
        # Try tuning this to values at or a little above
        # the mean of the positive target for the protected group
        # divided by the mean of the positive target for the unprotected group.
        # If it's set too large, the accuracy will be poor, so there
        # is a limit to the debiasing that can be obtained.
        self.mean_protected_prediction_ratio_minimum = 0.7
        ########################
        
        orig_cols = list(X.names)
              
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from collections import Counter
        
        import sys
        from aif360.sklearn.inprocessing import AdversarialDebiasing


        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, MaxAbsScaler
        from sklearn.metrics import accuracy_score


        import tensorflow as tf

              
        sess = tf.Session()
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                                 tmp_dir=self.context.tmp_dir,
                                                 experiment_tmp_dir=self.context.experiment_tmp_dir)
              

        # Target column
        self.target = 'high_priced'
        self.favorable_label = 0
        self.unfavorable_label = 1
        self.positive_target = self.favorable_label
        
        # Privleged_group_info  = [[Protetected group name 1, prevleged level, unprivleged level], [Protetected group name 2, prevleged level, unprivleged level]]
        # The protected group columns need to be binary
        protected_group_info = [['hispanic', 0, 1]]
        self.protected_name = protected_group_info[0][0]
        self.protected_label = protected_group_info[0][1]
        #########
        #########
        #########
        
        # Find the protected group column if it is present
        self.protected = "none"
        for col in orig_cols:
            split = col.split('_')
            if len(split) > 1:
                if (self.protected_name == col) or (self.protected_name == split[1]):
                    self.protected = col
                    protected_group_info[0][0] = self.protected 
            else:
                if (self.protected_name == col):
                    self.protected = col
                    protected_group_info[0][0] = self.protected 
                    
        loggerinfo(logger, "Protected test") 
        loggerinfo(logger, str(self.protected)) 
        loggerinfo(logger, str(split)) 
        loggerinfo(logger, str(self.protected_name)) 
        loggerinfo(logger, str(orig_cols))         
        
        # Set up protected group info
        self.protected_groups = [group_info[0] for group_info in protected_group_info]
       
 
        self.privileged_groups = []
        self.unprivileged_groups = []
        for protected_group in protected_group_info:
            
            self.privileged_groups_dict = {}
            self.unprivileged_groups_dict = {}
            self.privileged_groups_dict[protected_group[0]] = protected_group[1]
            self.unprivileged_groups_dict[protected_group[0]] = protected_group[2]
            self.privileged_groups.append(self.privileged_groups_dict)
            self.unprivileged_groups.append(self.unprivileged_groups_dict)

        loggerinfo(logger, "Protected B") 
        if self.protected != "none":
            # Set the protected group to 0 and all others 1          
            protected_train = [int(item) for item in ~(np.array(X[self.protected]) == self.protected_label)]
        else:
            self.params["adversary_loss_weight"]=0

            protected_train =  []


        


        # Fit weights on the full dataset to be used on the external test set, if given
        #RW_full = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        debiased_model = AdversarialDebiasing(prot_attr=self.protected_groups[0],
                                              scope_name='classifier',
                                              debias=True, 
                                              classifier_num_hidden_units= self.params["classifier_num_hidden_units"],
                                              batch_size=self.params["batch_size"],
                                              num_epochs = self.params["num_epochs"],
                                              adversary_loss_weight=self.params["adversary_loss_weight"])
        


        # Switch to pandas
        X = X.to_pandas()
        X.columns = orig_cols

    
                
        X_datatypes = [str(item) for item in list(X.dtypes)]
        
        # List the categorical and numerical features
        self.X_categorical = [orig_cols[col_count] for col_count in range(len(orig_cols)) if (X_datatypes[col_count] == 'category') or (X_datatypes[col_count] == 'object')]
        self.X_numeric = [item for item in orig_cols if item not in self.X_categorical]
        self.encoded_categories = []


        # Find the levels and mode for each categorical feature
        # for use in the test set
        self.train_levels = {}
        for item in self.X_categorical:
            self.train_levels[item] = list(set(X[item]))
            self.train_mode[item] = Counter(X[item]).most_common(1)[0][0] 
       
        
        # One hot encode the categorical features
        # And replace missing values with a Missing category
        if len(self.X_categorical) > 0:
            loggerinfo(logger, "Categorical encode")  
            
            for colname in self.X_categorical:
                X[colname] = list(X[colname].fillna("Missing")) 
                
            self.enc = OneHotEncoder(handle_unknown='ignore')

            if self.protected in self.X_categorical:
                self.X_categorical.remove(self.protected)
                
            if len(self.X_categorical) > 0:
                self.enc.fit(X[self.X_categorical])
                self.encoded_categories = list(self.enc.get_feature_names(input_features=self.X_categorical))

                X_enc=self.enc.transform(X[self.X_categorical]).toarray()

                X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1)


        # Replace missing values with a missing value code
        if len(self.X_numeric) > 0:
   
            for colname in self.X_numeric:
                X[colname] = list(X[colname].fillna(-999)) 


        loggerinfo(logger, "Columnname encode")  
        loggerinfo(logger, str(X.columns))  
        loggerinfo(logger, str(self.protected))  
        
        # Remove the protected value from the model
        X_final_cols = list(X.columns)
        if self.protected != "none":
                      
            X = X.drop(self.protected, axis=1)


        loggerinfo(logger, "Columnname encode4") 

        # Make sure the target that represents a positive outcome is 1
        if self.positive_target == 0:
            y = 1 - y
        X_full = X.copy()
        
        loggerinfo(logger, "Columnname encode5")     
        
        
        X_full.index = protected_train 
        X_full.index.name = self.protected
        loggerinfo(logger, str(X_full.shape))  
        loggerinfo(logger, str(y.shape)) 
        debiased_model.fit(X_full, y)

        loggerinfo(logger, "Columnname encode6")  

        self.mean_target = np.array(sum(y)/len(y))
       

        self.is_train = True
        
        
        loggerinfo(logger, "end")    
        loggerinfo(logger, str(list(X.columns)))
        loggerinfo(logger, str([1]*X.shape[1]))   
        loggerinfo(logger, str(self.params["adversary_loss_weight"]))   
        
        
        train_proba = debiased_model.predict_proba(X_full)
        
        loggerinfo(logger, "train p check")    
        loggerinfo(logger, str(train_proba))
        
        loggerinfo(logger, "final check")    
        loggerinfo(logger, str(len(orig_cols)))
        loggerinfo(logger, str(orig_cols))
        loggerinfo(logger, str(X_full.shape))
        loggerinfo(logger, str(X.shape))   
        loggerinfo(logger, str(X_final_cols))           
        loggerinfo(logger, str(len(X_final_cols)))       
        loggerinfo(logger, str([1]*len(X_final_cols)))  
        loggerinfo(logger, str(self.params["num_epochs"]))  

        
        imp = [random.random() for ii in range(len(X_final_cols))]


        # Set model properties
        self.set_model_properties(model=debiased_model,
                                  features=X_final_cols,
                                  importances=imp,
                                  iterations=self.params["num_epochs"])

           

    def predict(self, X, **kwargs):
        orig_cols = list(X.names)
        import pandas as pd
        import numpy as np
        from aif360.datasets import BinaryLabelDataset
        
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                                 tmp_dir=self.context.tmp_dir,
                                                 experiment_tmp_dir=self.context.experiment_tmp_dir)   

        loggerinfo(logger, "Predict start")   

            
        X = dt.Frame(X)
        
        X = X.to_pandas()
        
        loggerinfo(logger, "Test x")   
        loggerinfo(logger, str(orig_cols))   
        loggerinfo(logger, str(list(X.columns)))  

        
        if self.protected in list(X.columns):
            # Set the protected group to 0 and all others 1
            loggerinfo(logger, "Protected test found")   
            protected_test = np.array([int(item) for item in ~(np.array(X[self.protected]) == self.protected_label)])

        else:
            loggerinfo(logger, "Protected test not found")  
            protected_test = np.array([])
        
          
        # Replace missing values with a missing category
        # Replace categories that weren't in the training set with the mode
        if len(self.X_categorical) > 0:
            
            for colname in self.X_categorical:
                if colname in list(X.columns):
                    X[colname] = list(X[colname].fillna("Missing"))  
            
            for label in self.X_categorical:
                if label in list(X.columns):
                    # Replace anything not in the test set
                    train_categories = self.train_levels[label]
                    X_label = np.array(X[label])
                    mmode = self.train_mode[label]
                    X_label[~np.isin(X_label, train_categories)] = mmode
                    X[label] = X_label

        # Replace missing values with a missing value code    
        if len(self.X_numeric) > 0:
            for colname in self.X_numeric:
                if colname in list(X.columns):
                    X[colname] = list(X[colname].fillna(-999))            
                    
               
        # Get model    
        model, _, _, _ = self.get_model_properties()
        
        if self.protected in X:
            protected_values = list(X[self.protected])
            
   
        # Remove the protected group
        if self.protected in self.X_categorical:
            self.X_categorical.remove(self.protected)

        # One hot encode categorical features
        if len(self.X_categorical) > 0:
            X_enc=self.enc.transform(X[self.X_categorical]).toarray()
            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1) 
      

        X.index = protected_test
        preds = model.predict_proba(X)[:,1]
        # If the positive target was 0, change the final result to 1-p
        if self.positive_target == 0:
            preds = 1.0 - preds

            
        mean_preds = np.mean(preds)
        
        # Set a penalty value to which some probabilities will be changed
        # if the fairness threshold isn't reached
        epsilon = 0.0001
        if mean_preds > 0.5:
            penalty = epsilon
        else:
            penalty = 1.0 - epsilon
        
        # Only apply penalties in the training stage
        if self.is_train:      
            # If the protected value was removed, use the maximum penalty
            # by changing all probabilities to the penalty value
            # (the recipe needs to be able to use the protected values)
            if self.protected == "none":
               preds[0:len(preds)] = penalty        
               loggerinfo(logger, str(preds))    
               loggerinfo(logger, "Removal_penalty")    
                
            else:  
                # The mean ratio calculation for target=0 and target=1
                if self.positive_target == 0:
                    if np.mean(preds[protected_test == 1]) < 1.0:
                        DI = (1.0 - np.mean(preds[protected_test == 0]))/ (1.0 - np.mean(preds[protected_test == 1]))
                    else:
                        DI = 1
                else:
                    if np.mean(preds[protected_test == 1]) > 0.0:
                        DI = np.mean(preds[protected_test == 0]) / np.mean(preds[protected_test == 1])
                    else:
                        DI = 1                        

                    
                loggerinfo(logger, "Mean ratio Check")   
                loggerinfo(logger, str(DI))   
    
                if DI < self.mean_protected_prediction_ratio_minimum:
                    # Create a penalty proportional to the distance below the specified threshold
                    len_preds = len(preds)
                    num_penalty = min(len_preds, int((self.mean_protected_prediction_ratio_minimum-DI) / self.mean_protected_prediction_ratio_minimum * len_preds ))

                    preds[0:num_penalty] = penalty
                    loggerinfo(logger, "num_penalty1")                 
                    loggerinfo(logger, str(num_penalty), str(num_penalty/len(preds))) 
            
            
        self.is_train = False     

        return preds
