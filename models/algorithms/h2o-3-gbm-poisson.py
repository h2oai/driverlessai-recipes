"""H2O-3 Distributed Scalable Machine Learning Models: Poisson GBM
"""
from h2oaicore.models import CustomModel
import datatable as dt
import uuid
from h2oaicore.systemutils import config, user_dir, remove
from h2o.estimators.gbm import H2OGradientBoostingEstimator

import numpy as np

_global_modules_needed_by_name = ['h2o==3.34.0.7']
import h2o
import os


class H2OBaseModel:
    _regression = True
    _binary = False
    _multiclass = False
    _can_handle_non_numeric = True
    _can_handle_text = True  # but no special handling by base model, just doesn't fail
    _is_reproducible = False
    _check_stall = False  # avoid stall check. h2o runs as server, and is not a child for which we check CPU/GPU usage
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _class = NotImplemented

    @staticmethod
    def do_acceptance_test():
        return True  # Turn off to save time

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = None
        self.target = "__target__"
        self.weight = "__weight__"
        self.col_types = None
        self.my_log_dir = os.path.abspath(os.path.join(user_dir(),
                                                       config.contrib_relative_directory, "h2o_log"))
        if not os.path.isdir(self.my_log_dir):
            os.makedirs(self.my_log_dir, exist_ok=True)

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):

        self.params = dict(max_runtime_secs=0)

    def get_iterations(self, model):
        return 0

    def make_instance(self, **kwargs):
        return self.__class__._class(seed=self.random_state, **kwargs)

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)
        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = None

        orig_cols = list(X.names)
        train_X = h2o.H2OFrame(X.to_pandas())
        self.col_types = train_X.types
        train_y = h2o.H2OFrame(np.fabs(y),
                               column_names=[self.target],
                               column_types=['categorical' if self.num_classes >= 2 else 'numeric'])
        train_frame = train_X.cbind(train_y)
        if sample_weight is not None:
            train_w = h2o.H2OFrame(sample_weight,
                                   column_names=[self.weight],
                                   column_types=['numeric'])
            train_frame = train_frame.cbind(train_w)
        valid_frame = None
        valid_X = None
        valid_y = None
        model = None
        if eval_set is not None:
            valid_X = h2o.H2OFrame(eval_set[0][0].to_pandas(), column_types=self.col_types)
            valid_y = h2o.H2OFrame(np.fabs(eval_set[0][1]),
                                   column_names=[self.target],
                                   column_types=['categorical' if self.num_classes >= 2 else 'numeric'])
            valid_frame = valid_X.cbind(valid_y)
            if sample_weight is not None:
                if sample_weight_eval_set is None:
                    sample_weight_eval_set = [np.ones(len(eval_set[0][1]))]
                valid_w = h2o.H2OFrame(sample_weight_eval_set[0],
                                       column_names=[self.weight],
                                       column_types=['numeric'])
                valid_frame = valid_frame.cbind(valid_w)

        try:
            train_kwargs = dict()

            max_runtime_secs = self.params.get('max_runtime_secs', 0)
            train_kwargs = dict(max_runtime_secs=max_runtime_secs)

            if valid_frame is not None:
                train_kwargs['validation_frame'] = valid_frame
            if sample_weight is not None:
                train_kwargs['weights_column'] = self.weight
            model = self.make_instance(**self.params)
            model.train(x=train_X.names, y=self.target, training_frame=train_frame, **train_kwargs)
            self.id = model.model_id
            model_path = os.path.join(user_dir(), "h2o_model." + str(uuid.uuid4()))
            model_path = h2o.save_model(model=model, path=model_path)
            with open(model_path, "rb") as f:
                raw_model_bytes = f.read()

        finally:
            if model_path is not None:
                remove(model_path)
            for xx in [train_frame, train_X, train_y, model, valid_frame, valid_X, valid_y]:
                if xx is not None:
                    h2o.remove(xx)

        df_varimp = model.varimp(True)
        if df_varimp is None:
            varimp = np.ones(len(orig_cols))
        else:
            df_varimp.index = df_varimp['variable']
            df_varimp = df_varimp.iloc[:, 1]  # relative importance
            for missing in [x for x in orig_cols if x not in list(df_varimp.index)]:
                # h2o3 doesn't handle raw strings all the time, can hit:
                # KeyError: "None of [Index(['0_Str:secret_ChangeTemp'], dtype='object', name='variable')] are in the [index]"
                df_varimp[missing] = 0
            varimp = df_varimp[orig_cols].values  # order by fitted features
            varimp = np.nan_to_num(varimp)

        self.set_model_properties(model=raw_model_bytes,
                                  features=orig_cols,
                                  importances=varimp,
                                  iterations=self.get_iterations(model))

    def predict(self, X, **kwargs):
        model, _, _, _ = self.get_model_properties()
        X = dt.Frame(X)
        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = os.path.join(user_dir(), self.id)
        model_file = os.path.join(model_path, "h2o_model." + str(uuid.uuid4()) + ".bin")
        os.makedirs(model_path, exist_ok=True)
        with open(model_file, "wb") as f:
            f.write(model)
        model = h2o.load_model(os.path.abspath(model_file))
        test_frame = h2o.H2OFrame(X.to_pandas(), column_types=self.col_types)
        preds_frame = None

        try:
            preds_frame = model.predict(test_frame)
            preds = preds_frame.as_data_frame(header=False)

            return preds.values.ravel()

        finally:
            remove(model_file)
            # h2o.remove(self.id) # Cannot remove id, do multiple predictions on same model
            h2o.remove(test_frame)
            if preds_frame is not None:
                h2o.remove(preds_frame)


class H2OGBMPoissonModel(H2OBaseModel, CustomModel):
    _display_name = "H2O GBM Poisson"
    _description = "H2O-3 Gradient Boosting Machine with Poisson loss function."
    _class = H2OGradientBoostingEstimator

    def get_iterations(self, model):
        return model.params['ntrees']['actual'] + 1

    def mutate_params(self,
                      **kwargs):
        max_iterations = min(kwargs['n_estimators'],
                             config.max_nestimators) if 'n_estimators' in kwargs else config.max_nestimators
        max_iterations = min(kwargs['iterations'], max_iterations) if 'iterations' in kwargs else max_iterations
        self.params['ntrees'] = max_iterations
        self.params['stopping_rounds'] = int(np.random.choice([5, 10, 20]))
        self.params['learn_rate'] = max(1. / self.params['ntrees'], 0.005)
        self.params['max_depth'] = int(np.random.choice(range(2, 11)))
        self.params['col_sample_rate'] = float(np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.params['sample_rate'] = float(np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.params['distribution'] = "poisson"
