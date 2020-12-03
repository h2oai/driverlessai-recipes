"""Generalized Additive Model"""

import uuid
import os
import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count
from h2oaicore.systemutils import user_dir, remove, config, IgnoreError
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdebug


class GAM(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False
    _display_name = "GAM"
    _description = "Generalized Additive Model"
    _modules_needed_by_name = ['shap', 'collections', 'scipy', 'pandas', 'matplotlib', 'sklearn', 'pygam']
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def do_acceptance_test():
        return True

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = dict(random_state=kwargs.get("random_state", 1234),
                           max_depth_duplication=None, n_estimators=10,
                           lam=0.1, max_iter=100)

    def mutate_params(self, accuracy=10, **kwargs):

        if accuracy > 8:
            lam = [0, 0.001, 0.01, 0.1, 1.0, 3.0, 5.0, 10.0]
            max_iter = [100, 1000]

        elif accuracy >= 5:
            lam = [0, 0.01, 0.1, 1.0, 10.0]
            max_iter = [100]

        else:
            lam = [0, 0.01, 0.1, 1.0, 10.0]
            max_iter = [100]

        self.params["lam"] = np.random.choice(lam)
        self.params["max_iter"] = np.random.choice(max_iter)

    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files 
        # Set the default value without context available (required to pass acceptance test)
        tmp_folder = os.path.join(user_dir(), "%s_GAM_model_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_GAM_model_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "GAM was denied temp folder creation rights")
            tmp_folder = os.path.join(user_dir(), "%s_GAM_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "GAM temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_GAM_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(user_dir(), "%s_GAM_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "GAM temp folder {}".format(tmp_folder))
        return tmp_folder

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):

        orig_cols = list(X.names)

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from collections import Counter
        import pygam
        from pygam import LinearGAM, LogisticGAM
        import matplotlib.pyplot as plt

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

            clf = LogisticGAM(terms="auto", lam=self.params["lam"], max_iter=self.params["max_iter"])
            self.is_classifier = True

        else:
            clf = LinearGAM(terms="auto", lam=self.params["lam"], max_iter=self.params["max_iter"])
            self.is_classifier = False

        X = self.basic_impute(X)
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
        self.X_categorical = [orig_cols[col_count] for col_count in range(len(orig_cols)) if
                              (X_datatypes[col_count] == 'category') or (X_datatypes[col_count] == 'object')]
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

            X_enc = self.enc.transform(X[self.X_categorical]).toarray()

            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1)

        # Replace missing values with a missing value code
        self.median_train = {}

        if len(self.X_numeric) > 0:
            for colname in self.X_numeric:
                self.median_train[colname] = X[colname].quantile(0.5)
                X.loc[:, colname] = X[colname].fillna(self.median_train[colname]).copy()

        try:
            clf.fit(X, y)
        except np.linalg.LinAlgError as e:
            raise IgnoreError("np.linalg.LinAlgError") from e
        except pygam.utils.OptimizationError as e:
            raise IgnoreError("pygam.utils.OptimizationError") from e
        except ValueError as e:
            if 'On entry to DLASCL parameter number' in str(e):
                raise IgnoreError('On entry to DLASCL parameter number') from e
            raise

        p_values = np.array(clf.statistics_['p_values'])

        # Plot the partial dependence plots for each feature
        for ii in range(X.shape[1]):
            XX = clf.generate_X_grid(term=ii)
            plt.figure();
            plt.plot(XX[:, ii], clf.partial_dependence(term=ii, X=XX))
            plt.plot(XX[:, ii], clf.partial_dependence(term=ii, X=XX, width=.95)[1], c='r', ls='--')
            plt.title("Partial Dependence " + X.columns[ii], fontdict={'fontsize': 10})
            plt.show()
            plt.savefig(os.path.join(tmp_folder, 'Feature_partial_dependence_' + str(X.columns[ii])[0:10] + '.png'),
                        bbox_inches="tight")

        if max(p_values[0:(len(p_values) - 1)]) > 0:
            importances = -np.log(p_values[0:(len(p_values) - 1)] + 10 ** (-16))

            importances = list(importances / max(importances))
        else:
            importances = [1] * (len(p_values) - 1)

        self.mean_target = np.array(sum(y) / len(y))

        self.set_model_properties(model=clf,
                                  features=list(X.columns),
                                  importances=importances,
                                  iterations=self.params['n_estimators'])

    def basic_impute(self, X):
        # scikit extra trees internally converts to np.float32 during all operations,
        # so if float64 datatable, need to cast first, in case will be nan for float32
        from h2oaicore.systemutils import update_precision
        X = update_precision(X, data_type=np.float32, override_with_data_type=True, fixup_almost_numeric=True)
        # Replace missing values with a value smaller than all observed values
        if not hasattr(self, 'min'):
            self.min = dict()
        for col in X.names:
            XX = X[:, col]
            if col not in self.min:
                self.min[col] = XX.min1()
                if self.min[col] is None or np.isnan(self.min[col]) or np.isinf(self.min[col]):
                    self.min[col] = -1e10
                else:
                    self.min[col] -= 1
            XX.replace([None, np.inf, -np.inf], self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        return X

    def predict(self, X, **kwargs):
        orig_cols = list(X.names)
        import pandas as pd

        X = dt.Frame(X)
        X = self.basic_impute(X)

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
            for colname in self.X_numeric:
                self.median_train[colname] = X[colname].quantile(0.5)
                X.loc[:, colname] = X[colname].fillna(self.median_train[colname]).copy()

                # Get model    
        model, _, _, _ = self.get_model_properties()

        # One hot encode categorical features
        if len(self.X_categorical) > 0:
            X_enc = self.enc.transform(X[self.X_categorical]).toarray()
            X = pd.concat([X[self.X_numeric], pd.DataFrame(X_enc, columns=self.encoded_categories)], axis=1)

            # Make predictions on the test set
        if self.is_classifier:
            p = model.predict_proba(X)
        else:
            p = model.predict(X)

        p[np.isnan(p)] = self.mean_target

        return p
