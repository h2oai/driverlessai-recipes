# Purpose: For transactional grouped data, calculate historical percentage of the target value
#
# Expected ML Problem: Transactional binary classification for many groups
# Example Use Cases: part failure, credit card fraud
# Overview: Takes hardcoded groups and time
#
# DAI Version: 1.8.0
# Important Note: Models with this transformer should be retrained frequently


from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
from datetime import timedelta
from h2oaicore.systemutils import (
    small_job_pool, save_obj, load_obj, temporary_files_path, remove, max_threads, config,
    make_experiment_logger, loggerinfo, loggerwarning
)
import os
import uuid
import shutil
import joblib


# Parallel implementation requires methods being called from different processes
# Global methods support this feature
# We use global methods as a wrapper for member methods of the transformer
def transformer_fit_transform_async(*args, **kwargs):
    return ParallelPCTDaysTransformer._fit_transform_async(*args, **kwargs)


class ParallelPCTDaysTransformer(CustomTransformer):
    # only for binary classification
    _regression = False
    _binary = True
    _multiclass = False

    _numeric_output = True
    _is_reproducible = True
    _included_model_classes = None
    _excluded_model_classes = None

    # give easy-to-understand names in the UI
    _allow_transform_to_modify_output_feature_names = True

    # Allow transformer to be used by DAI
    @staticmethod
    def is_enabled():
        return True

    # Cannot use DAI code testing as we have hard coded column names
    @staticmethod
    def do_acceptance_test():
        return False

    # Requires all columns in the data set
    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1)

    # Set the hard coded values about our data set
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.col_date = "event_ts"
        self.col_group = ["customer_id"]
        self.target_positive = "Yes"

        # the number of DAYS or Rows we want to look back
        self.timespans = [30, 60, 90]

    # This function runs during training
    def fit_transform(self, X: dt.Frame, y: np.array = None, **kwargs):

        # Get logger, create temp folder for parallel processing
        # and get number of workers allocated to the transformer
        logger = self.get_experiment_logger()
        tmp_folder = self._create_tmp_folder(logger)
        n_jobs = self._get_n_jobs(logger, **kwargs)

        # values we need to accessible in other parts of DAI
        features = []
        self._output_feature_names = []
        self._feature_desc = []

        # list of group by and time columns
        cols_agg = self.col_group.copy()
        cols_agg.append(self.col_date)

        # list of group by, time, and y columns
        cols_agg_y = cols_agg.copy()
        cols_agg_y.append('col_target_bin')

        # Move to pandas
        # Pandas index is no 0 to len X - 1
        X = X.to_pandas()

        # Make a 0/1 column for if the event happened or not
        X['col_target_bin'] = (y == self.target_positive).astype(int)

        # Make sure time is time
        X[self.col_date] = pd.to_datetime(X[self.col_date])

        # We need to sort by date then groupby customer_id and finally shift 1 event
        X = X.sort_values(self.col_date)
        X['col_target_bin'] = X.groupby(self.col_group)['col_target_bin'].shift(1).fillna(0)
        # Go back to original index
        X.sort_index(inplace=True)

        # Create a feature for each of the requested days / rows
        # Initialize the feature result with index set, this way features are added in correct row order even
        # if feature comes out with an unordered index
        feats_df = pd.DataFrame(index=X.index)
        for t in self.timespans:
            # Create feature name
            t_days = str(t) + "d"  # use t instead of t_days if you want to aggregate by rows - other units of time can be used as well
            # Create the groups
            groups = X[cols_agg_y].groupby(self.col_group)

            # prepare parallel processing
            df_paths = []

            def processor(out, res):
                out.append(res)

            num_tasks = len(groups)
            pool_to_use = small_job_pool
            pool = pool_to_use(
                logger=None, processor=processor,
                num_tasks=num_tasks, max_workers=n_jobs)

            # Go though the groups and predict only top
            feat_list = []
            for _key, _df in groups:
                # Sort the group by date, this is required for the rolling per time interval to work
                _df = _df.sort_values(self.col_date)
                # Create path for storage
                a_path = os.path.join(tmp_folder, "grp_prev_dev_df" + str(uuid.uuid4()))
                # Store frame
                save_obj(_df, a_path)
                # Prepare arguments to be passed to the fit transform method
                args = (a_path, self.col_date, t_days, tmp_folder)
                kwargs = {}
                # Add to the pool
                pool.submit_tryget(
                    logger=None,
                    function=transformer_fit_transform_async,
                    args=args, kwargs=kwargs,
                    out=df_paths  # df_paths will receive the path where job results are stored
                )

            # Now that the jobs have been processed
            # Concat group results and assign to result dataframe
            # Since dataframe is indexed, feature is inluded in the appropriate row_order
            # i.e. pandas make sure index on both sides agree
            pool.finish()
            feats_df[t_days] = pd.concat((load_obj(a_path) for a_path in df_paths), axis=0)

            for p in df_paths:
                remove(p)

            # Set the column names and descriptions
            self._output_feature_names.append("Y%: " + t_days)
            self._feature_desc.append("Percent of Target with Event in Last " + str(t) + " Days")

        # keep the max days needed of transactions for each person to look up in transform
        time_filter = (
            X[cols_agg].groupby(self.col_group)[self.col_date].transform('max') - X[self.col_date]
        ) <= timedelta(days=max(self.timespans))

        self.lookup_df = X.loc[time_filter].sort_values(self.col_date)

        # Clean temp folder before exiting fit_transform
        self._clean_tmp_folder(logger, tmp_folder)

        # return the features
        return feats_df

    @staticmethod
    def _fit_transform_async(df_path, col_date, t_days, tmp_folder):
        # Load data
        _df = load_obj(df_path)
        # Apply rolling window but make sure target does not contain any NaN
        res = _df.set_index(col_date)['col_target_bin'].fillna(0).rolling(t_days, min_periods=1).mean()
        # It is not an apply and index may not be the original one
        # Therefore make sure the index is the original one
        res.index = _df.index
        # Store result
        res_path = os.path.join(tmp_folder, "grp_prev_days_res_df" + str(uuid.uuid4()))
        save_obj(res, res_path)
        remove(df_path)  # remove to indicate success
        return res_path

    # This function runs during scoring - it will look to what happened in the training data, so may get stale fast
    # If we see a new customer we return 0 for the features, as there are no key events in their past
    # It is also used to transform the validation set,
    # which means validation set must be after (time wise) the training set
    def transform(self, X: dt.Frame, **kwargs):

        # Get logger, create temp folder for parallel processing
        # and get number of workers allocated to the transformer
        logger = self.get_experiment_logger()
        tmp_folder = self._create_tmp_folder(logger)
        n_jobs = self._get_n_jobs(logger, **kwargs)

        # group by and time
        cols_agg = self.col_group.copy()
        cols_agg.append(self.col_date)

        # group by, time, and y
        cols_agg_y = cols_agg.copy()
        cols_agg_y.append('col_target_bin')

        # pandas
        X = X.to_pandas()
        original_index = X.index

        # Make sure time is time
        X[self.col_date] = pd.to_datetime(X[self.col_date])

        # combine rows from training and new rows, col_target_bin will be null in new data
        lkup_and_test = pd.concat([self.lookup_df, X], axis=0, sort=False).reset_index()

        feats_df = pd.DataFrame(index=lkup_and_test.index)
        for t in self.timespans:
            # Create feature name
            t_days = str(t) + "d"
            # Create the groups
            groups = lkup_and_test[cols_agg_y].groupby(self.col_group)

            # prepare parallel processing
            df_paths = []

            def processor(out, res):
                out.append(res)

            num_tasks = len(groups)
            pool_to_use = small_job_pool
            pool = pool_to_use(
                logger=None, processor=processor,
                num_tasks=num_tasks, max_workers=n_jobs)

            # Run through groups
            feat_list = []
            for _key, _df in groups:
                # Sort the group by date, this is required for the rolling per time interval to work
                _df = _df.sort_values(self.col_date)
                # Create path for storage
                a_path = os.path.join(tmp_folder, "grp_prev_days_df" + str(uuid.uuid4()))
                # Store frame
                save_obj(_df, a_path)
                # Prepare arguments to be passed to the fit transform method
                args = (a_path, self.col_date, t_days, tmp_folder)
                kwargs = {}
                # Add to the pool
                pool.submit_tryget(
                    logger=None,
                    function=transformer_fit_transform_async,
                    args=args, kwargs=kwargs,
                    out=df_paths  # df_paths will receive the path where job results are stored
                )

                # Now that the jobs have been processed
                # Concat group results and assign to result dataframe
                # Since dataframe is indexed, feature is inluded in the appropriate row_order
                # i.e. pandas make sure index on both sides agree
            pool.finish()
            feats_df[t_days] = pd.concat((load_obj(a_path) for a_path in df_paths), axis=0)

            for p in df_paths:
                remove(p)

        # Only return rows whose index is below self.lookup_df size
        feats_df = feats_df.loc[self.lookup_df.shape[0]:]
        feats_df.index = original_index

        # Clean temp folder before exiting fit_transform
        self._clean_tmp_folder(logger, tmp_folder)

        return feats_df

    def get_experiment_logger(self):
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )
        return logger

    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files used during multi processing experiment
        # This temp folder will be removed at the end of the process
        # Set the default value without context available (required to pass acceptance test
        tmp_folder = os.path.join(temporary_files_path, "%s_grp_prev_days_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_grp_prev_days_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "Transformer was denied temp folder creation rights")
            tmp_folder = os.path.join(temporary_files_path, "%s_grp_prev_days_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "Transformer temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_grp_prev_days_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(temporary_files_path, "%s_grp_prev_days_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "Transformer temp folder {}".format(tmp_folder))
        return tmp_folder

    def _clean_tmp_folder(self, logger, tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
            loggerinfo(logger, "Transformer cleaned up temporary file folder.")
        except:
            loggerwarning(logger, "Transformer could not delete the temporary file folder.")

    @staticmethod
    def _get_n_jobs(logger, **kwargs):
        try:
            if config.fixed_num_folds <= 0:
                n_jobs = max(1, int(int(max_threads() / min(config.num_folds, kwargs['max_workers']))))
            else:
                n_jobs = max(1, int(
                    int(max_threads() / min(config.fixed_num_folds, config.num_folds, kwargs['max_workers']))))

        except KeyError:
            loggerinfo(logger, "Transformer No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        loggerinfo(logger, f"Transformer n_jobs {n_jobs}")

        return n_jobs