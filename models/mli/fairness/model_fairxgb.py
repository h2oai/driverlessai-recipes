"""Fair XGB """

import uuid
import os
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import user_dir, max_threads
from h2oaicore.systemutils import make_experiment_logger, loggerdata, loggerwarning, loggerdebug, loggerinfo


class FAIRXGBOOST(CustomModel):
    _regression = False
    _binary = True
    _multiclass = False
    _display_name = "Fair_XGBOOST"
    _description = "Fair_XGBOOST"

    @staticmethod
    def do_acceptance_test():
        return False

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):

        self.params = dict(random_state=kwargs.get("random_state", 24),
                           eta=0.1, max_depth=12, min_child_weight=2.0,
                           reg_lambda=1.0, colsample_bytree=0.8,
                           subsample=1.0, mu=0.1, reg_alpha=0,
                           n_jobs=self.params_base.get('n_jobs', max_threads()),
                           )

    def mutate_params(self, accuracy=10, **kwargs):
        if accuracy > 8:
            eta = [0.5, 0.1, 0.05, 0.01]
            max_depth = list(range(4, 21))
            min_child_weight = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
            reg_lambda = [0.0, 0.1, 1.0, 2.0, 5.0, 8.0, 10.0, 20.0]
            reg_alpha = [0.0, 0.1, 1.0, 5.0, 10.0]
            colsample_bytree = [0.1 * ii for ii in range(1, 11)]
            subsample = [0.5, 0.8, 0.9, 1.0]
            mu = [0.05 * ii for ii in range(1, 14)]

        elif accuracy >= 5:
            eta = [0.5, 0.1, 0.05]
            max_depth = list(range(4, 21, 2))
            min_child_weight = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
            reg_lambda = [0.0, 0.1, 1.0, 2.0, 5.0, 8.0, 10.0, 20.0]
            reg_alpha = [0.0, 0.1, 1.0]
            colsample_bytree = [0.1 * ii for ii in range(2, 11, 2)]
            subsample = [1.0]
            mu = [0.05 * ii for ii in range(1, 14)]

        else:
            eta = [0.1]
            max_depth = list(range(4, 21, 2))
            min_child_weight = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
            reg_lambda = [0.0, 0.1, 1.0, 5.0, 10.0]
            reg_alpha = [0.0]
            colsample_bytree = [0.1 * ii for ii in range(2, 11, 2)]
            subsample = [1.0]
            mu = [0.05 * ii for ii in range(1, 14)]

        self.params["eta"] = np.random.choice(eta)
        self.params["max_depth"] = np.random.choice(max_depth)
        self.params["min_child_weight"] = np.random.choice(min_child_weight)
        self.params["reg_lambda"] = np.random.choice(reg_lambda)
        self.params["reg_alpha"] = np.random.choice(reg_alpha)
        self.params["colsample_bytree"] = np.random.choice(colsample_bytree)
        self.params["subsample"] = np.random.choice(subsample)
        self.params["mu"] = np.random.choice(mu)
        self.params["n_jobs"] = self.params_base.get('n_jobs', max_threads())

    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files 
        # Set the default value without context available (required to pass acceptance test)
        tmp_folder = os.path.join(user_dir(), "%s_FAIRXGB_model_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_FAIRXGB_model_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "FAIRXGB was denied temp folder creation rights")
            tmp_folder = os.path.join(user_dir(), "%s_FAIRXGB_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "FAIRXGB temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_FAIRXGB_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(user_dir(), "%s_FAIRXGB_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerdata(logger, "FAIRXGB temp folder {}".format(tmp_folder))
        return tmp_folder

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):

        # Specify these parameters for the dataset.
        #
        # Also set feature engineering effort to 0
        # under the features section of expert settings.
        ########################
        # Specify the protected column.
        # The protected column must be numeric.
        self.protected_name = "black"
        # Specify the level of the protected group in the protected column
        self.protected_label = 1
        # Specify the target level considered to be a positive outcome
        # Must be encoded as 0/1
        self.positive_target = 0
        # Set minimum mean protected ratio needed to avoid a penalty 
        # (mean protected ratio = mean predictions for the protected group/mean predictions for all other groups)
        #
        # Try tuning this to values at or a little above
        # the mean of the positive target for the protected group
        # divided by the mean of the positive target for the unprotected group.
        # If it's set too large, the accuracy will be poor, so there
        # is a limit to the debiasing that can be obtained.
        self.mean_protected_prediction_ratio_minimum = 0.92
        ########################

        orig_cols = list(X.names)

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from collections import Counter
        import xgboost as xgb

        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                            tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)

        # Current mu value
        mu = self.params["mu"]

        def fair_metric(predt: np.ndarray, dtrain: xgb.DMatrix):
            ''' FairXGB Error Metric'''
            # predt is the prediction array 

            # Find the right protected group vector
            if len(predt) == len(protected_train):
                protected_feature = np.array(protected_train.copy())

            elif len(predt) == len(protected_full):
                protected_feature = np.array(protected_full.copy())

            elif len(predt) == len(protected_valid):
                protected_feature = np.array(protected_valid.copy())

            else:
                protected_feature = 0

            y = dtrain.get_label()

            answer = - y * np.log(sigmoid(predt)) - (1 - y) * np.log(1 - sigmoid(predt))

            answer += mu * (protected_feature * np.log(sigmoid(predt)) + (1 - protected_feature) * np.log(
                1 - sigmoid(predt)))

            return 'Fair_Metric', float(np.sum(answer) / len(answer))

        def sigmoid(x):
            z = 1.0 / (1.0 + np.exp(-x))
            return z

        def gradient(predt: np.ndarray, dtrain: xgb.DMatrix):
            '''Fair Xgboost Gradient'''
            # predt is the prediction array 

            # Find the right protected group vector            
            if len(predt) == len(protected_train):
                protected_feature = np.array(protected_train.copy())

            elif len(predt) == len(protected_full):
                protected_feature = np.array(protected_full.copy())

            elif len(predt) == len(protected_valid):
                protected_feature = np.array(protected_valid.copy())

            else:
                protected_feature = 0

            y = dtrain.get_label()

            answer = sigmoid(predt) - y
            answer += mu * (protected_feature - sigmoid(predt))

            return answer

        def hessian(predt: np.ndarray, dtrain: xgb.DMatrix):
            '''Fair Xgboost Hessian'''
            # predt is the prediction array 

            answer = (1 - mu) * sigmoid(predt) * (1 - sigmoid(predt))

            return answer

        def fair(predt: np.ndarray, dtrain: xgb.DMatrix):
            ''' Fair xgb objective function
            '''

            grad = gradient(predt, dtrain)
            hess = hessian(predt, dtrain)
            return grad, hess

            # Set up model

        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)

            params = {}
            params['eta'] = self.params["eta"]
            params['max_depth'] = self.params['max_depth']
            params['min_child_weight'] = self.params['min_child_weight']
            params['reg_lambda'] = self.params['reg_lambda']
            params['reg_alpha'] = self.params['reg_alpha']
            params['colsample_bytree'] = self.params['colsample_bytree']
            params['subsample'] = self.params['subsample']
            params['silent'] = 1
            params['seed'] = self.params['random_state']
        else:
            # fairxgb doesn't work for regression
            loggerinfo(logger, "PASS, no fairxgboost model")
            pass

        # Switch to pandas
        X = X.to_pandas()
        X.columns = orig_cols

        # Find the protected group column if it is present
        self.protected = "none"
        for col in X.columns:
            if col.find(self.protected_name) > -1:
                self.protected = col

        X_datatypes = [str(item) for item in list(X.dtypes)]

        # List the categorical and numerical features
        self.X_categorical = [orig_cols[col_count] for col_count in range(len(orig_cols)) if
                              (X_datatypes[col_count] == 'category') or (X_datatypes[col_count] == 'object')]
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

                X_enc = self.enc.transform(X[self.X_categorical]).toarray()

                X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1)

        # Replace missing values with a missing value code
        if len(self.X_numeric) > 0:

            for colname in self.X_numeric:
                X[colname] = list(X[colname].fillna(-999))

                # Make sure the target that represents a positive outcome is 1
        if self.positive_target == 0:
            y = 1 - y
        X_full = X.copy()
        y_full = y.copy()

        # Set up a validation step to find the optimal number of trees
        X_valid = X.iloc[int(0.7 * len(X_full)):, :]
        y_valid = y[int(0.7 * len(X_full)):]
        X = X.iloc[0:int(0.7 * len(X_full)), :]
        y = y[0:int(0.7 * len(X_full))]

        if self.protected != "none":
            # Set the protected group to 0 and all others 1          
            protected_full = [int(item) for item in ~(np.array(X_full[self.protected]) == self.protected_label)]
            protected_train = [int(item) for item in ~(np.array(X[self.protected]) == self.protected_label)]
            protected_valid = [int(item) for item in ~(np.array(X_valid[self.protected]) == self.protected_label)]
        else:
            mu = 0
            protected_full = []
            protected_train = []
            protected_valid = []

        # Remove the protected value from the model
        if self.protected != "none":
            X = X.drop(self.protected, axis=1)

            X_full = X_full.drop(self.protected, axis=1)

            X_valid = X_valid.drop(self.protected, axis=1)

        d_train = xgb.DMatrix(X, label=y, missing=np.nan)

        d_valid = xgb.DMatrix(X_valid, label=y_valid, missing=np.nan)

        # Initial run to find the optimal number of trees
        num_iterations = 10000
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]

        clf = xgb.train(params, d_train, num_iterations, watchlist, feval=fair_metric, verbose_eval=10, obj=fair,
                        early_stopping_rounds=10)

        # Second xgboost run with the full dataset and optimal number of trees
        attribute_dict = clf.attributes()
        new_iterations = int(attribute_dict['best_iteration'])

        d_train = xgb.DMatrix(X_full, label=y_full, missing=np.nan)
        watchlist = [(d_train, 'train')]
        clf = xgb.train(params, d_train, new_iterations, watchlist, feval=fair_metric, verbose_eval=10, obj=fair)

        # Calculate feature importances
        importances_dict = clf.get_score(importance_type='gain')

        # Make sure the protected group has high feature importance
        # so that it doesn't get dropped by driverless
        if self.protected != "none":
            if len(importances_dict) > 0:
                importances_dict[self.protected] = max(importances_dict.values())
            else:

                importances_dict[self.protected] = 1
                for col in list(X.columns):
                    importances_dict[col] = 1

        # Make sure any dropped columns are listed with 0 importance
        for col in list(X.columns):
            if col not in importances_dict:
                importances_dict[col] = 0

        self.mean_target = np.array(sum(y) / len(y))

        loggerinfo(logger, "End fair check")
        loggerinfo(logger, str(mu))
        loggerdata(logger, str(importances_dict))
        self.is_train = True

        # Set model properties
        self.set_model_properties(model=clf,
                                  features=list(importances_dict.keys()),
                                  importances=list(importances_dict.values()),
                                  iterations=num_iterations)

    def predict(self, X, **kwargs):
        orig_cols = list(X.names)
        import pandas as pd
        import xgboost as xgb
        import numpy as np

        def sigmoid(x):
            z = 1.0 / (1.0 + np.exp(-x))
            return z

        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id,
                                            tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)

        X = dt.Frame(X)

        X = X.to_pandas()

        if self.protected in list(X.columns):
            # Set the protected group to 0 and all others 1
            loggerdebug(logger, "Protected test found")
            protected_test = np.array([int(item) for item in ~(np.array(X[self.protected]) == self.protected_label)])

        else:
            loggerdebug(logger, "Protected test not found")
            protected_test = np.array([])

        if self.protected in list(X.columns):
            X = X.drop(self.protected, axis=1)

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

        # Remove the protected group
        if self.protected in self.X_categorical:
            self.X_categorical.remove(self.protected)

        # One hot encode categorical features
        if len(self.X_categorical) > 0:
            X_enc = self.enc.transform(X[self.X_categorical]).toarray()
            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1)

        d_test = xgb.DMatrix(X, missing=np.nan)

        # If the positive target was 0, change the final result to 1-p
        if self.positive_target == 0:
            preds = 1.0 - sigmoid(model.predict(d_test))
        else:
            preds = sigmoid(model.predict(d_test))

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
                loggerdata(logger, str(preds))
                loggerdata(logger, "Removal_penalty")

            else:
                # The mean ratio calculation for target=0 and target=1
                if self.positive_target == 0:
                    if np.mean(preds[protected_test == 1]) < 1.0:
                        DI = (1.0 - np.mean(preds[protected_test == 0])) / (1.0 - np.mean(preds[protected_test == 1]))
                    else:
                        DI = 1
                else:
                    if np.mean(preds[protected_test == 1]) > 0.0:
                        DI = np.mean(preds[protected_test == 0]) / np.mean(preds[protected_test == 1])
                    else:
                        DI = 1

                loggerdata(logger, "Mean ratio Check")
                loggerdata(logger, str(DI))

                if DI < self.mean_protected_prediction_ratio_minimum:
                    # Create a penalty proportional to the distance below the specified threshold
                    len_preds = len(preds)
                    num_penalty = min(len_preds, int((
                                                                 self.mean_protected_prediction_ratio_minimum - DI) / self.mean_protected_prediction_ratio_minimum * len_preds))

                    preds[0:num_penalty] = penalty
                    loggerdata(logger, "num_penalty1")
                    loggerdata(logger, str(num_penalty), str(num_penalty / len(preds)))

        self.is_train = False

        return preds
