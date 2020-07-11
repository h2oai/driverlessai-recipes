"""Decision tree plus linear model"""


import uuid
import os
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count
from h2oaicore.systemutils import temporary_files_path, remove, config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdebug


class DECISION_TREE_PLUS_LINEAR(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False
    _display_name = "DECISION_TREE_PLUS_LINEAR"
    _description = "Takes the results of a decision tree and then fits a linear model to each set of node data"
    _modules_needed_by_name = ['shap', 'collections', 'scipy', 'pandas', 
                               'matplotlib', 'sklearn']

    @staticmethod
    def do_acceptance_test():
        return True

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = dict(random_state=kwargs.get("random_state", 1234),
                           max_depth_duplication = None, n_estimators= 10,
                           tree_depth=3)

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            tree_depth = [4, 5]
        elif accuracy >= 5:
            tree_depth = [3, 4]
        else:
            tree_depth = [2]

        self.params["tree_depth"] = np.random.choice(tree_depth) 


    def _create_tmp_folder(self, logger):
        # Create a temp folder to store xnn files 
        # Set the default value without context available (required to pass acceptance test)
        tmp_folder = os.path.join(temporary_files_path, "%s_DTL_model_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_DTL_model_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "DTL was denied temp folder creation rights")
            tmp_folder = os.path.join(temporary_files_path, "%s_DTL_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "DTL temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_DTL_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(temporary_files_path, "%s_DTL_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "DTL temp folder {}".format(tmp_folder))
        return tmp_folder


    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from collections import Counter
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn import tree
        import matplotlib.pyplot as plt
              
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                                 tmp_dir=self.context.tmp_dir,
                                                 experiment_tmp_dir=self.context.experiment_tmp_dir)
            
        # Set up temp folter
        tmp_folder = self._create_tmp_folder(logger)        
        
        # Set up model
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            
            clf = DecisionTreeClassifier(random_state=42, max_depth=self.params["tree_depth"])
            self.is_classifier = True
            
        else:
            clf = DecisionTreeRegressor(random_state=42, max_depth=self.params["tree_depth"])
            self.is_classifier = False
        
        # Find the datatypes
        X = X.to_pandas()
        X.columns = orig_cols

        # Change continuous features to categorical
        X_datatypes = [str(item) for item in list(X.dtypes)]
        
        # Change all float32 values to float64
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
            
            X.loc[:, self.X_categorical] = X[self.X_categorical].fillna("Missing").copy()
            self.enc = OneHotEncoder(handle_unknown='ignore')

            self.enc.fit(X[self.X_categorical])
            self.encoded_categories = list(self.enc.get_feature_names(input_features=self.X_categorical))

            X_enc=self.enc.transform(X[self.X_categorical]).toarray()

            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1)
            
        # Replace missing values with a missing value code
        if len(self.X_numeric) > 0:
            X.loc[:, self.X_numeric] = X[self.X_numeric].fillna(-999).copy()   
            
        clf.fit(X, y)
        if self.is_classifier:
            yy = clf.predict_proba(X)
        
            p = np.round_(yy[:, 1], 5)
        else:
            yy = clf.predict(X)
        
            p = np.round_(yy, 5)
            
        self.leaf_categories = list(set(p))
        
        model_array = {}
        equation_log = []
        for cat in self.leaf_categories:
            if self.is_classifier:
                if (np.mean(y[p==cat])<1) and (np.mean(y[p==cat])>0):
                        
                    lm = LogisticRegression(random_state=42)

                    lm.fit(X[p==cat], y[p==cat])
                    
                    model_array[cat] = lm
                    equation_log.append([[int(round((1-cat)*sum(p==cat))), int(round(cat*sum(p==cat)))], sum(p==cat), lm.intercept_[0]] +list(lm.coef_[0]))
                else:
                    loggerinfo(logger, "No leaf fit")
                    model_array[cat] = "dt"
            else:
                try:
                    lm = LinearRegression()
                    lm.fit(X[p==cat], y[p==cat])
                    
                    model_array[cat] = lm
                    
                    equation_log.append([cat, sum(p==cat), lm.intercept_] +list(lm.coef_))
                except:
                    loggerinfo(logger, "No leaf fit")
                    model_array[cat] = "dt"                    
                
        pd.DataFrame(equation_log, columns=['leaf value', 'number of samples', 'intercept'] + list(X.columns)).to_csv(os.path.join(tmp_folder, 'Leaf_model_coef.csv'))

        fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (8, 8), dpi=1600)
        tree.plot_tree(clf, feature_names = list(X.columns))
        fig.savefig(os.path.join(tmp_folder, 'Decision_tree_plot.png'))
        
        importances = clf.feature_importances_
        loggerinfo(logger, str(importances))
        
        self.mean_target = np.array(sum(y)/len(y))        
        
        model = [clf, model_array]
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
        X = X.to_pandas()

        X_datatypes = [str(item) for item in list(X.dtypes)]
        
        # Change float 32 values to float 64
        for ii in range(len(X_datatypes)):
            if X_datatypes[ii] == 'float32':
               X = X.astype({orig_cols[ii]: np.float64})      
               
        # Replace missing values with a missing category
        # Replace categories that weren't in the training set with the mode
        if len(self.X_categorical) > 0:
            
            X.loc[:, self.X_categorical] = X[self.X_categorical].fillna("Missing").copy()
            

            for label in self.X_categorical:
                # Replace anything not in the test set
                train_categories = self.train_levels[label]
                X_label = np.array(X[label])
                mmode = self.train_mode[label]
                X_label[~np.isin(X_label, train_categories)] = mmode
                X[label] = X_label
        
        # Replace missing values with a missing value code    
        if len(self.X_numeric) > 0:
            X.loc[:, self.X_numeric] = X[self.X_numeric].fillna(-999).copy()               
    
        # Get model    
        model, _, _, _ = self.get_model_properties()
   

        
        # One hot encode categorical features
        if len(self.X_categorical) > 0:
            X_enc=self.enc.transform(X[self.X_categorical]).toarray()
            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1) 
      
        # Make predictions on the test set

        if self.is_classifier:
            y = model[0].predict_proba(X)
            p = np.round_(y[:,1], 5)
        else:
            y = model[0].predict(X)
            p = np.round_(y, 5)
            
        pp = p.copy()
        
        for cat in self.leaf_categories:
            if len(X[p==cat]) > 0:
                if model[1][cat] != "dt":
                
                    lm = model[1][cat]
                    if self.is_classifier:
                        temp = lm.predict_proba(X[p==cat])
                        pp[p==cat] = temp[:, 1]  
                    else:
                        temp = lm.predict(X[p==cat])  
                        pp[p==cat] = temp 
                     
        pp[np.isnan(pp)] = self.mean_target
                        
        return pp
