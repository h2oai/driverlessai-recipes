"""H2O-3 Distributed Scalable Machine Learning Models (DL/GLM/GBM/DRF/NB/AutoML)
"""
import copy

from h2oaicore.models import CustomModel
import datatable as dt
import uuid
from h2oaicore.systemutils import config, user_dir, remove
import numpy as np

_global_modules_needed_by_name = ['h2o==3.30.0.3']
import h2o
import os


class H2OBaseModel:
    _regression = True
    _binary = True
    _multiclass = True
    _can_handle_non_numeric = True
    _is_reproducible = False  # since using max_runtime_secs - disable that if need reproducible models
    _check_stall = False  # avoid stall check. h2o runs as server, and is not a child for which we check CPU/GPU usage
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _class = NotImplemented

    @staticmethod
    def do_acceptance_test():
        return False  # save time

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
        max_runtime_secs = 600
        if accuracy is not None and time_tolerance is not None:
            max_runtime_secs = accuracy * (time_tolerance + 1) * 10  # customize here to your liking
        self.params = dict(max_runtime_secs=max_runtime_secs)

    def get_iterations(self, model):
        return 0

    def make_instance(self, **kwargs):
        return self.__class__._class(seed=self.random_state, **kwargs)

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = None

        if isinstance(self, H2ONBModel):
            # NB can only handle weights of 0 / 1
            if sample_weight is not None:
                sample_weight = (sample_weight != 0).astype(int)
            if sample_weight_eval_set is not None:
                sample_weight_eval_set = [(sample_weight_eval_set[0] != 0).astype(int)]

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
            params = copy.deepcopy(self.params)
            if not isinstance(self, H2OAutoMLModel):
                # AutoML needs max_runtime_secs in initializer, all others in train() method
                max_runtime_secs = params.pop('max_runtime_secs')
                train_kwargs = dict(max_runtime_secs=max_runtime_secs)
            if valid_frame is not None:
                train_kwargs['validation_frame'] = valid_frame
            if sample_weight is not None:
                train_kwargs['weights_column'] = self.weight
            model = self.make_instance(**params)

            # Don't ever use the offset column as a feature
            offset_col = None  # if no column is called offset we will pass "None" and not use this feature
            cols_to_train = []  # list of all non-offset columns

            for col in list(train_X.names):
                if not col.lower() == "offset":
                    cols_to_train.append(col)
                else:
                    offset_col = col

            orig_cols = cols_to_train  # not training on offset

            # Models that can use an offset column
            if isinstance(model, H2OGBMModel) | isinstance(model, H2ODLModel) | isinstance(model, H2OGLMModel):
                model.train(x=cols_to_train, y=self.target, training_frame=train_frame, offset_column=offset_col,
                            **train_kwargs)
            else:
                model.train(x=train_X.names, y=self.target, training_frame=train_frame, **train_kwargs)

            if isinstance(model, H2OAutoML):
                model = model.leader
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
                    if isinstance(xx, H2OAutoML):
                        h2o.remove(xx.project_name)
                    else:
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
            if kwargs.get("pred_contribs"):
                return model.predict_contributions(test_frame).as_data_frame(header=False).values
            preds_frame = model.predict(test_frame)
            preds = preds_frame.as_data_frame(header=False)
            if self.num_classes == 1:
                return preds.values.ravel()
            elif self.num_classes == 2:
                return preds.iloc[:, -1].values.ravel()
            else:
                return preds.iloc[:, 1:].values
        finally:
            # h2o.remove(self.id) # Cannot remove id, do multiple predictions on same model
            h2o.remove(test_frame)
            remove(model_file)
            if preds_frame is not None:
                h2o.remove(preds_frame)


from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator


class H2ONBModel(H2OBaseModel, CustomModel):
    _regression = False

    _display_name = "H2O NB"
    _description = "H2O-3 Naive Bayes"
    _class = H2ONaiveBayesEstimator

    def predict(self, X, **kwargs):
        preds = super().predict(X, **kwargs)
        return np.nan_to_num(preds, copy=False)  # get rid of infs


from h2o.estimators.gbm import H2OGradientBoostingEstimator


class H2OGBMModel(H2OBaseModel, CustomModel):
    _display_name = "H2O GBM"
    _description = "H2O-3 Gradient Boosting Machine"
    _class = H2OGradientBoostingEstimator

    @property
    def has_pred_contribs(self):
        return self.labels is None or len(self.labels) <= 2

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


from h2o.estimators.random_forest import H2ORandomForestEstimator


class H2ORFModel(H2OBaseModel, CustomModel):
    _display_name = "H2O RF"
    _description = "H2O-3 Random Forest"
    _class = H2ORandomForestEstimator

    @property
    def has_pred_contribs(self):
        return self.labels is None or len(self.labels) <= 2

    def get_iterations(self, model):
        return model.params['ntrees']['actual'] + 1

    def mutate_params(self,
                      **kwargs):
        max_iterations = min(kwargs['n_estimators'],
                             config.max_nestimators) if 'n_estimators' in kwargs else config.max_nestimators
        max_iterations = min(kwargs['iterations'], max_iterations) if 'iterations' in kwargs else max_iterations
        self.params['ntrees'] = max_iterations
        self.params['stopping_rounds'] = int(np.random.choice([5, 10, 20]))
        self.params['max_depth'] = int(np.random.choice(range(2, 11)))
        self.params['sample_rate'] = float(np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))


from h2o.estimators.deeplearning import H2ODeepLearningEstimator


class H2ODLModel(H2OBaseModel, CustomModel):
    _is_reproducible = False

    _display_name = "H2O DL"
    _description = "H2O-3 DeepLearning"
    _class = H2ODeepLearningEstimator

    def mutate_params(self,
                      accuracy=10, time_tolerance=10,
                      **kwargs):
        self.params['activation'] = np.random.choice(["rectifier", "rectifier",  # upweight
                                                      "rectifier_with_dropout",
                                                      "tanh"])
        self.params['hidden'] = np.random.choice([[20, 20, 20],
                                                  [50, 50, 50],
                                                  [100, 100, 100],
                                                  [200, 200], [200, 200, 200],
                                                  [500], [500, 500], [500, 500, 500]])
        self.params['epochs'] = accuracy * max(1, time_tolerance)
        self.params['input_dropout_ratio'] = float(np.random.choice([0, 0.1, 0.2]))


from h2o.estimators.glm import H2OGeneralizedLinearEstimator


class H2OGLMModel(H2OBaseModel, CustomModel):
    _display_name = "H2O GLM"
    _description = "H2O-3 Generalized Linear Model"
    _class = H2OGeneralizedLinearEstimator

    def make_instance(self, **kwargs):
        if self.num_classes == 1:
            return self.__class__._class(seed=self.random_state, family='gaussian')  # tweedie/poisson/tweedie/gamma
        elif self.num_classes == 2:
            return self.__class__._class(seed=self.random_state, family='binomial')
        else:
            return self.__class__._class(seed=self.random_state, family='multinomial')


from h2o.automl import H2OAutoML


class H2OAutoMLModel(H2OBaseModel, CustomModel):
    @staticmethod
    def is_enabled():
        return False  # automl inside automl can be too slow, especially given small max_runtime_secs above

    @staticmethod
    def do_acceptance_test():
        return False  # save time

    _display_name = "H2O AutoML"
    _description = "H2O-3 AutoML"
    _class = H2OAutoML
