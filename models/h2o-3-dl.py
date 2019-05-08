from h2oaicore.models import CustomModel
import datatable as dt
import _pickle as pickle
import uuid
from h2oaicore.systemutils import config, temporary_files_path



import h2o
import os
from h2o.estimators.deeplearning import H2ODeepLearningEstimator


class H2ODLModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _can_handle_text = True
    _is_reproducible = False

    _boosters = ['h2odl']
    _display_name = "H2O DL"
    _description = "H2O-3 DeepLearning"

    def __init__ (self, **kwargs):
        super().__init__(**kwargs)
        self.id = None
        self.target = "__target__"

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)
        h2o.init(port=config.h2o_recipes_port)
        model_path = None

        orig_cols = list(X.names)
        train_X = h2o.H2OFrame(X.to_pandas())
        train_y = h2o.H2OFrame(y,
                               column_names=[self.target],
                               column_types=['categorical' if self.num_classes >= 2 else 'numeric'])
        train_frame = train_X.cbind(train_y)
        valid_frame = None
        valid_X = None
        valid_y = None
        model = None
        if eval_set is not None:
            valid_X = h2o.H2OFrame(eval_set[0][0].to_pandas())
            valid_y = h2o.H2OFrame(eval_set[0][1],
                                   column_names=[self.target],
                                   column_types=['categorical' if self.num_classes >= 2 else 'numeric'])
            valid_frame = valid_X.cbind(valid_y)

        try:
            model = H2ODeepLearningEstimator()
            model.train(x=train_X.names,
                        y=self.target,
                        training_frame=train_frame,
                        validation_frame=valid_frame)
            self.id = model.model_id
            model_path = os.path.join(temporary_files_path, "h2o_model." + str(uuid.uuid4()))
            model_path = h2o.save_model(model=model, path=model_path)
            with open(model_path, "rb") as f:
                raw_model_bytes = f.read()

        finally:
            if model_path is not None:
                os.remove(model_path)
            for xx in [train_frame, train_X, train_y, model, valid_frame, valid_X, valid_y]:
                if xx is not None:
                    h2o.remove(xx)

        df_varimp = model.varimp(True)
        df_varimp.index = df_varimp['variable']
        df_varimp = df_varimp.iloc[:, 1]  # relative importance
        df_varimp = df_varimp[orig_cols]  # order by fitted features

        self.set_model_properties(model=raw_model_bytes,
                                  features=orig_cols,
                                  importances=df_varimp.values,
                                  iterations=0)
        return self

    def predict(self, X, **kwargs):
        model, _, _, _ = self.get_model_properties()
        X = dt.Frame(X)
        h2o.init(port=config.h2o_recipes_port)
        model_path = os.path.join(temporary_files_path, self.id)
        with open(model_path, "wb") as f:
            f.write(model)
        model = h2o.load_model(model_path)
        os.remove(model_path)
        test_frame = h2o.H2OFrame(X.to_pandas())
        preds_frame = None

        pred_contribs = kwargs.get('pred_contribs', None)
        output_margin = kwargs.get('output_margin', None)

        try:
            if not pred_contribs:
                preds_frame = model.predict(test_frame)
                preds = preds_frame.as_data_frame(header=False)
                if self.num_classes == 1:
                    return preds.values.ravel()
                elif self.num_classes == 2:
                    return preds.iloc[:, 2].values.ravel()
                else:
                    return preds.iloc[:, 1:].values
            else:
                raise NotImplementedError("Latest H2O-3 has Shapley - call predict_contribs")
        finally:
            h2o.remove(self.id)
            h2o.remove(test_frame)
            if preds_frame is not None:
                h2o.remove(preds_frame)
