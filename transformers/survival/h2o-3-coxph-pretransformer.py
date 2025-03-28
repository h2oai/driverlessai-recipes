"""Pre-transformer utilizing survival analysis modeling using CoxPH (Cox proportional hazard)
   using H2O-3 CoxPH function.
   It adds risk score produced by CoxPH model and drops stop_column feature used for
   survival modeling along with actual target as event."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import uuid

_global_modules_needed_by_name = ['h2o==3.46.0.7']
import h2o
from h2oaicore.systemutils import temporary_files_path, config, remove
from h2o.estimators.coxph import H2OCoxProportionalHazardsEstimator
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
from h2oaicore.separators import extra_prefix, orig_feat_prefix


class SurvivalCoxPHPreTransformer(CustomTransformer):
    # only works with binomial problem for now
    _regression = False
    _binary = True
    _multiclass = False
    _numeric_output = False
    _can_be_pretransformer = True
    _default_as_pretransformer = True
    _must_be_pretransformer = True
    _only_as_pretransformer = True
    _unsupervised = False  # uses target
    _uses_target = True  # uses target

    # Duration (stop) column name
    _stop_column_name = "surv_days"
    _ignored_columns = None
    _survival_event = '__event__'

    def __init__(self, context=None, ties="breslow", max_iterations=20, **kwargs):
        super().__init__(context=context, **kwargs)
        self.ties = ties
        self.max_iterations = max_iterations
        self.id = None
        self.raw_model_bytes = None
        self.my_log_dir = os.path.abspath(os.path.join(temporary_files_path,
                                                       config.contrib_relative_directory, "h2o_log"))
        if not os.path.isdir(self.my_log_dir):
            os.makedirs(self.my_log_dir, exist_ok=True)

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {
            "ties": ["breslow", "efron"],
            "max_iterations": [10, 20, 30]
        }

    def fit_transform(self, X: dt.Frame, y: np.array = None, **kwargs):

        X_original = X

        X = X[:, dt.f[int].extend(dt.f[float]).extend(dt.f[bool]).extend(dt.f[str])]

        if hasattr(self, 'runcount'):
            self.run_count += 1
        else:
            self.run_count = 0

        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        survival_event = self.__class__._survival_event
        if survival_event in X.names:
            raise ValueError("Consider renaming feature '{}'.".format(survival_event))

        # bind y to X to use as event in CoxPH
        X[:, survival_event] = np.array(LabelEncoder().fit_transform(y))

        # sanity check that target is binary
        if X[survival_event].nunique()[0, 0] != 2:
            raise ValueError(
                "Too many values {} in event column - must be exactly 2.".format(X[survival_event].nunique()[0, 0]))

        # redress target values into 0, 1
        event_max = X[survival_event].max()[0, 0]
        X[dt.f[survival_event] != event_max, survival_event] = 0
        X[dt.f[survival_event] == event_max, survival_event] = 1

        stop_column_name = self.__class__._stop_column_name
        ignored_columns = self.__class__._ignored_columns

        if stop_column_name is None:
            raise ValueError("Stop column name can't be null.")

        main_message = "Survival Analysis CoxPH pre-transformer will use event '{}' and time '{}' columns.". \
            format(survival_event, stop_column_name)

        # in accpetance test simply return input X
        if stop_column_name not in X.names:
            loggerwarning(logger,
                          "Survival Analysis CoxPH pre-transformer found no time column '{}'.".format(stop_column_name))
            return X_original

        if not X[:, stop_column_name].stype in [dt.bool8, dt.int8, dt.int16, dt.int32, dt.int64, dt.float32,
                                                dt.float64]:
            raise ValueError("Stop column `{}' type must be numeric, but found '{}'".
                             format(stop_column_name, X[:, stop_column_name].stype))

        # remove stop column from X
        del X_original[:, stop_column_name]

        self._output_feature_names = list(X_original.names)
        self._feature_desc = list(X_original.names)

        if self.run_count == 0 and self.context and self.context.experiment_id:
            loggerinfo(logger, main_message)
            task = kwargs.get('task')
            if task and main_message is not None:
                task.sync(key=self.context.experiment_id, progress=dict(type='update', message=main_message))
                task.flush()

        # Validate CoxPH requirements on stop column
        if X[stop_column_name].min()[0, 0] < 0:
            X[dt.f[stop_column_name] < 0, stop_column_name] = 0
            loggerwarning(logger, "Stop column can't be negative: replaced negative values with 0.")
        if X[stop_column_name].countna()[0, 0] > 0:
            X[dt.isna(dt.f[stop_column_name]), stop_column_name] = 0
            loggerwarning(logger, "Stop column can't contain NULLs: replaced NULL with 0.")

        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model = H2OCoxProportionalHazardsEstimator(stop_column=stop_column_name,
                                                   ties=self.ties,
                                                   max_iterations=self.max_iterations)
        frame = h2o.H2OFrame(X.to_pandas())
        model_path = None
        risk_frame = None
        try:
            model.train(y=survival_event, training_frame=frame, ignored_columns=ignored_columns)
            self.id = model.model_id
            model_path = os.path.join(temporary_files_path, "h2o_model." + str(uuid.uuid4()))
            model_path = h2o.save_model(model=model, path=model_path)
            with open(model_path, "rb") as f:
                self.raw_model_bytes = f.read()
            risk_frame = model.predict(frame)
            X_original[:, "risk_score_coxph_{}_{}".format(self.ties, self.max_iterations)] = risk_frame.as_data_frame(
                header=False)
            self._output_feature_names.append(
                f"{self.display_name}{orig_feat_prefix}riskscore_coxph{extra_prefix}{self.ties}_{self.max_iterations}")
            self._feature_desc.append(f"CoxPH model risk score [ties={self.ties}, max.iter={self.max_iterations}")
            return X_original
        finally:
            if model_path is not None:
                remove(model_path)
            h2o.remove(model)
            h2o.remove(frame)
            if risk_frame is not None:
                h2o.remove(risk_frame)

    def transform(self, X: dt.Frame):

        stop_column_name = self.__class__._stop_column_name
        if stop_column_name in X.names:
            del X[:, stop_column_name]
        else:
            return X

        if self.id is None:
            return X

        # self._output_feature_names = list(X.names)
        # self._feature_desc = list(X.names)

        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = os.path.join(temporary_files_path, self.id)
        with open(model_path, "wb") as f:
            f.write(self.raw_model_bytes)
        model = h2o.load_model(os.path.abspath(model_path))
        remove(model_path)

        frame = h2o.H2OFrame(X.to_pandas())
        try:
            risk_frame = model.predict(frame)
            X[:, "risk_score_coxph_{}_{}".format(self.ties, self.max_iterations)] = risk_frame.as_data_frame(
                header=False)
            return X
        finally:
            h2o.remove(self.id)
            h2o.remove(frame)
            if risk_frame is not None:
                h2o.remove(risk_frame)
