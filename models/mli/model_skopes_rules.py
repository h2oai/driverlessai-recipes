"""Skopes rules """

import uuid
import os
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count
from h2oaicore.systemutils import temporary_files_path, remove, config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdebug


class SKOPE_RULES(CustomModel):
    _regression = False
    _binary = True
    _multiclass = False
    _display_name = "SKOPE RULES"
    _description = "SKOPE RULES"
    _modules_needed_by_name = ['shap', 'collections', 'scipy', 'pandas', 'matplotlib', 'sklearn', 'skope-rules']

    @staticmethod
    def do_acceptance_test():
        return True

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = dict(random_state=kwargs.get("random_state", 1234),
                           max_depth_duplication = None, n_estimators= 10,
                           precision_min = 0.5, recall_min=0.01, max_samples = 0.8,
                           max_samples_features = 1.0, max_depth = 3,
                           max_features = "auto", min_samples_split = 2, 
                           bootstrap= False, bootstrap_features = False)

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            max_depth_duplication = [None, 2, 3]
            n_estimators= [10, 20, 40]
            precision_min = [0.1, 0.2, 0.3]
            recall_min=[0.01, 0.05]
            max_samples = [0.5, 0.8, 1.0]
            max_samples_features = [0.5, 0.8, 1.0]

            max_depth = [3, 4, 5]
            max_features = ["sqrt", "log2", "auto"]
            min_samples_split = [2, 11, 21]
            bootstrap= [True, False]     
            bootstrap_features = [True, False]  
        elif accuracy >= 5:
            max_depth_duplication = [None]
            n_estimators= [10, 20]
            precision_min = [0.1, 0.2, 0.3]
            recall_min=[0.01]
            max_samples = [0.8, 1.0]
            max_samples_features = [1.0]

            max_depth = [3, 4]
            max_features = ["sqrt", "log2", "auto"]
            min_samples_split = [2, 5, 11]
            bootstrap= [True, False]     
            bootstrap_features = [True, False]  
        else:
            max_depth_duplication = [None]
            n_estimators= [10]
            precision_min = [0.1, 0.2]
            recall_min=[0.01]
            max_samples = [0.8, 1.0]
            max_samples_features = [0.8, 1.0]

            max_depth = [3, 4]
            max_features = ["auto"]
            min_samples_split = [2]
            bootstrap= [True, False]     
            bootstrap_features = [True, False]  
        # Modify certain parameters for tuning
        self.params["max_depth_duplication"] = np.random.choice(max_depth_duplication)
        self.params["n_estimators"] = np.random.choice(n_estimators)
        self.params["precision_min"] = np.random.choice(precision_min)        
        self.params["recall_min"] = np.random.choice(recall_min)        
        self.params["max_samples"] = np.random.choice(max_samples) 
        self.params["max_samples_features"] = np.random.choice(max_samples_features) 
        self.params["max_depth"] = np.random.choice(max_depth) 
        self.params["max_features"] = np.random.choice(max_features) 
        self.params["min_samples_split"] = np.random.choice(min_samples_split) 
        self.params["bootstrap"] = np.random.choice(bootstrap) 
        self.params["bootstrap_features"] = np.random.choice(bootstrap_features) 


    def _create_tmp_folder(self, logger):
        # Create a temp folder to store xnn files 
        # Set the default value without context available (required to pass acceptance test)
        tmp_folder = os.path.join(temporary_files_path, "%s_SKOPE_model_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_SKOPE_model_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "SKOPE was denied temp folder creation rights")
            tmp_folder = os.path.join(temporary_files_path, "%s_SKOPE_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "SKOPE temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_SKOPE_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(temporary_files_path, "%s_SKOPE_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "SKOPE temp folder {}".format(tmp_folder))
        return tmp_folder


    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
       
        
        import pandas as pd
        import numpy as np
        from skrules import SkopeRules
        from sklearn.preprocessing import OneHotEncoder
        from collections import Counter

              
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                                 tmp_dir=self.context.tmp_dir,
                                                 experiment_tmp_dir=self.context.experiment_tmp_dir)
            
        # Set up temp folder
        tmp_folder = self._create_tmp_folder(logger)        
        
        # Set up model
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            
            model = SkopeRules(max_depth_duplication=self.params["max_depth_duplication"] ,
                               n_estimators=self.params["n_estimators"],
                               precision_min=self.params["precision_min"],
                               recall_min=self.params["recall_min"],
                               max_samples=self.params["max_samples"],
                               max_samples_features=self.params["max_samples_features"],
                               max_depth=self.params["max_depth"],
                               max_features=self.params["max_features"],
                               min_samples_split=self.params["min_samples_split"],
                               bootstrap=self.params["bootstrap"],
                               bootstrap_features=self.params["bootstrap_features"],
                               random_state=self.params["random_state"],
                               feature_names=orig_cols)
        else:
            # Skopes doesn't work for regression
            loggerinfo(logger, "PASS, no skopes model")  
            pass
        
        
        # Find the datatypes
        X = X.to_pandas()

        # Change continuous features to categorical
        X_datatypes = [str(item) for item in list(X.dtypes)]
        
        # Change all float32 values to float64
        # Skopes crashes otherwise
        for ii in range(len(X_datatypes)):
            if X_datatypes[ii] == 'float32':
               X = X.astype({orig_cols[ii]: np.float64})
                
        X_datatypes = [str(item) for item in list(X.dtypes)]
        
        # List the categorical and numerical features
        self.X_categorical = [orig_cols[col_count] for col_count in range(len(orig_cols)) if (X_datatypes[col_count] == 'category') or (X_datatypes[col_count] == 'object')]
        self.X_numeric = [item for item in orig_cols if item not in self.X_categorical]

        # Find the levels and mode for each categorical feature
        # for use in the test set
        self.train_levels = {}
        for item in self.X_categorical:
            self.train_levels[item] = list(set(X[item]))
            self.train_mode[item] = Counter(X[item]).most_common(1)[0][0] 

        # One hot encode the categorical features
        # And replace missing values with a Missing category
        if len(self.X_categorical) > 0:
            loggerinfo(logger, "PCategorical encode")  
            
            #X.loc[:, self.X_categorical] = X[self.X_categorical].fillna("Missing").copy()
            for colname in self.X_categorical:
                X[colname] = list(X[colname].fillna("Missing")) 
            self.enc = OneHotEncoder(handle_unknown='ignore')

            self.enc.fit(X[self.X_categorical])
            self.encoded_categories = list(self.enc.get_feature_names(input_features=self.X_categorical))

            X_enc=self.enc.transform(X[self.X_categorical]).toarray()

            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1)

        # Replace missing values with a missing value code
        if len(self.X_numeric) > 0:
            #X.loc[:, self.X_numeric] = X[self.X_numeric].fillna(-999).copy()      
            for colname in self.X_numeric:
                X[colname] = list(X[colname].fillna(-999)) 
        
        model.fit(np.array(X), np.array(y))
    
        # Find the rule list
        self.rule_list = model.rules_

        # Calculate feature importances
        var_imp = []
        for var in orig_cols:
            var_imp.append(sum(int(var in item[0]) for item in self.rule_list))

        if max(var_imp) != 0:
            importances = list(np.array(var_imp)/max(var_imp))
        else:
            importances = [1] * len(var_imp)   
            
        pd.DataFrame(model.rules_).to_csv(os.path.join(tmp_folder, 'Skope_rules.csv'), index=False)
 
        self.mean_target = np.array(sum(y)/len(y))
       
        # Set model properties
        self.set_model_properties(model=model,
                                  features=list(X.columns),
                                  importances=importances,
                                  iterations=self.params['n_estimators'])
        

    def predict(self, X, **kwargs):
        orig_cols = list(X.names)
        import pandas as pd
        
        X = dt.Frame(X)
        
        # Find datatypes
        X=X.to_pandas()

        X_datatypes = [str(item) for item in list(X.dtypes)]
        
        # Change float 32 values to float 64
        for ii in range(len(X_datatypes)):
            if X_datatypes[ii] == 'float32':
               X = X.astype({orig_cols[ii]: np.float64})      
               
        # Replace missing values with a missing category
        # Replace categories that weren't in the training set with the mode
        if len(self.X_categorical) > 0:
            
            for colname in self.X_categorical:
                X[colname] = list(X[colname].fillna("Missing"))  
            
            for label in self.X_categorical:
                # Replace anything not in the test set
                train_categories = self.train_levels[label]
                X_label = np.array(X[label])
                mmode = self.train_mode[label]
                X_label[~np.isin(X_label, train_categories)] = mmode
                X[label] = X_label

        # Replace missing values with a missing value code    
        if len(self.X_numeric) > 0:
            for colname in self.X_numeric:
                X[colname] = list(X[colname].fillna(-999))            
              
        # Get model    
        model, _, _, _ = self.get_model_properties()
   
        # One hot encode categorical features
        if len(self.X_categorical) > 0:
            X_enc=self.enc.transform(X[self.X_categorical]).toarray()
            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1) 
      
        # Make predictions on the test set
        preds=model.score_top_rules(X) / len(self.rule_list)
        preds=np.array(preds)
        epsilon = 10**(-3)
        preds = np.nan_to_num(preds, nan=self.mean_target)
        preds[preds>1-epsilon] = 1.0 - epsilon
        preds[preds<0+epsilon] = 0.0 + epsilon

        return preds
