"""Decision Forest Model based on Intel DAAL"""

import datatable as dt
from h2oaicore.models import CustomModel
import daal4py as d4p
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DaalForestModel(CustomModel):
    _regression = False
    _binary = True
    _multiclass = False
    _display_name = "DaalForest"
    _description = "Decision Forest Model based on Intel DAAL (https://intelpython.github.io/daal4py/algorithms.html)"

    def set_default_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params = dict(nClasses=self.num_classes,
                           fptype='float',
                           varImportance='MDI',
                           nTrees=10)

    def mutate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        pass

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        lb = LabelEncoder()
        lb.fit(self.labels)
        y = lb.transform(y)

        if isinstance(X, dt.Frame):
            X_features = list(X.names)
            X = X.to_numpy()
            dtype = np.float32 if self.params['fptype'] == 'float' else np.float64
            X = np.ascontiguousarray(X, dtype=dtype)
            y = np.ascontiguousarray(y, dtype=dtype).reshape(X.shape[0], 1)
        else:
            raise

        train_algo = d4p.decision_forest_classification_training(**self.params)
        train_result = train_algo.compute(X, y)
        model_tuple = (train_result, self.num_classes, self.params['fptype'])
        importances = train_result.variableImportance.tolist()[0]
        self.set_model_properties(model=model_tuple,
                                  features=X_features,
                                  importances=importances,
                                  iterations=self.params['nTrees'])

    def predict(self, X, **kwargs):
        model_tuple, _, _, _ = self.get_model_properties()
        train_result = model_tuple[0]
        nClasses = model_tuple[1]
        fptype = model_tuple[2]
        predict_algo = d4p.decision_forest_classification_prediction(nClasses=nClasses, fptype=fptype)
        X = X.to_numpy()
        result = predict_algo.compute(X, train_result.model).prediction.ravel()
        return result
