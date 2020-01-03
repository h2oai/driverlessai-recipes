"""H2O-3 Distributed Scalable Machine Learning Models: Poisson GLM
"""
from h2oaicore.models import CustomModel
import datatable as dt
import uuid
from h2oaicore.systemutils import config, temporary_files_path, remove
import numpy as np

_global_modules_needed_by_name = ['h2o==3.26.0.1']
import h2o
import os


class H2OBaseModel:
    _regression = True
    _binary = False
    _multiclass = False
    _can_handle_non_numeric = True
    _is_reproducible = False
    _check_stall = False  # avoid stall check. h2o runs as server, and is not a child for which we check CPU/GPU usage

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
        self.my_log_dir = os.path.abspath(os.path.join(config.data_directory,
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
        train_y = h2o.H2OFrame(y,
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
            valid_y = h2o.H2OFrame(eval_set[0][1],
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

            max_runtime_secs = self.params.pop('max_runtime_secs')
            train_kwargs = dict(max_runtime_secs=max_runtime_secs)

            if valid_frame is not None:
                train_kwargs['validation_frame'] = valid_frame
            if sample_weight is not None:
                train_kwargs['weights_column'] = self.weight
            model = self.make_instance(**self.params)
            model.train(x=train_X.names, y=self.target, training_frame=train_frame, **train_kwargs)
            self.id = model.model_id
            model_path = os.path.join(temporary_files_path, "h2o_model." + str(uuid.uuid4()))
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
            varimp = df_varimp[orig_cols].values  # order by fitted features

        self.set_model_properties(model=raw_model_bytes,
                                  features=orig_cols,
                                  importances=varimp,
                                  iterations=self.get_iterations(model))

    def predict(self, X, **kwargs):
        model, _, _, _ = self.get_model_properties()
        X = dt.Frame(X)
        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = os.path.join(temporary_files_path, self.id)
        with open(model_path, "wb") as f:
            f.write(model)
        model = h2o.load_model(os.path.abspath(model_path))
        remove(model_path)
        test_frame = h2o.H2OFrame(X.to_pandas(), column_types=self.col_types)
        preds_frame = None

        try:
            preds_frame = model.predict(test_frame)
            preds = preds_frame.as_data_frame(header=False)

            return preds.values.ravel()

        finally:
            h2o.remove(self.id)
            h2o.remove(test_frame)
            if preds_frame is not None:
                h2o.remove(preds_frame)


from h2o.estimators.glm import H2OGeneralizedLinearEstimator


class H2OGLMPoissonModel(H2OBaseModel, CustomModel):
    _display_name = "H2O GLM Poisson"
    _description = "H2O-3 Generalized Linear Model"
    _class = H2OGeneralizedLinearEstimator

    def mutate_params(self,
                      **kwargs):
        self.params['family'] = "poisson"
        self.params['link'] = np.random.choice(["log", "identity"])
