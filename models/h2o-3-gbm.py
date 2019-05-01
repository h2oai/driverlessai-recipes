from h2oaicore.models import CustomModel
import h2o
import os
from h2o.estimators.gbm import H2OGradientBoostingEstimator


class H2OGBMModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True

    _boosters = ['h2ogbm']
    _display_name = "H2O GBM"
    _description = "H2O-3 Gradient Boosting Machine"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = None
        self.raw_model_bytes = None
        self.target = "__target__"

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)
        h2o.init()
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
            model = H2OGradientBoostingEstimator()
            model.train(x=train_X.names,
                        y=self.target,
                        training_frame=train_frame,
                        validation_frame=valid_frame)
            self.id = model.model_id
            model_path = h2o.save_model(model=model)
            with open(model_path, "rb") as f:
                self.raw_model_bytes = f.read()

        finally:
            if model_path is not None:
                os.remove(model_path)
            for xx in [train_frame, train_X, train_y, model, valid_frame, valid_X, valid_y]:
                if xx is not None:
                    h2o.remove(xx)

        # need to move to wrapper
        self.orig_cols = orig_cols
        self.feature_names_fitted = orig_cols
        self.transformed_features = self.feature_names_fitted
        self.best_ntree_limit = model.params['ntrees']['actual']
        # must always set best_iterations
        self.best_iterations = self.best_ntree_limit + 1

        df_varimp = model.varimp(True)
        df_varimp.index = df_varimp['variable']
        df_varimp = df_varimp.iloc[:, 1]  # relative importance
        df_varimp = df_varimp[self.feature_names_fitted]  # order by fitted features
        self.set_feature_importances(df_varimp.values)
        return self

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        h2o.init()
        with open(self.id, "wb") as f:
            f.write(self.raw_model_bytes)
        model = h2o.load_model(self.id)
        os.remove(self.id)
        test_frame = h2o.H2OFrame(X.to_pandas())
        preds_frame = None

        pred_contribs = kwargs.get('pred_contribs', None)
        output_margin = kwargs.get('output_margin', None)

        try:
            if not pred_contribs:
                preds_frame = model.predict(test_frame)
                preds = preds_frame.as_data_frame(header=False).values
                if self.num_classes == 1:
                    return preds.ravel()
                elif self.num_classes == 2:
                    return preds[:, -1].ravel()
                else:
                    return preds[:, 1:]
            else:
                raise NotImplementedError("H2O-3 has Shapley, just needs to be implemented")
        finally:
            h2o.remove(self.id)
            h2o.remove(test_frame)
            if preds_frame is not None:
                h2o.remove(preds_frame)

