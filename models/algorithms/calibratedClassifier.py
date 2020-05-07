""" Calibrated Classifier Model: To calibrate predictions using Platt's scaling or Isotonic regression
"""

import copy
import datatable as dt
from h2oaicore.systemutils import config
from h2oaicore.models import CustomModel, LightGBMModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit

    
class CalibratedClassifierModel:
    _regression = False
    _binary = True
    _multiclass = False
    _can_use_gpu = True
    _mojo = False
    _description = "Calibrated Classifier Model (LightGBM)"
    
    le = LabelEncoder()

    @staticmethod
    def do_acceptance_test():
        """
        Return whether to do acceptance tests during upload of recipe and during start of Driverless AI.
        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return True

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        assert len(self.__class__.__bases__) == 3
        assert CalibratedClassifierModel in self.__class__.__bases__
        
        self.le.fit(self.labels)
        y_ = self.le.transform(y)
        
        sss = StratifiedShuffleSplit(n_splits = 1, test_size = self.params["calib_perc"], random_state=4235)
        tr_indx, te_indx = next(iter(sss.split(y_.reshape(-1,1), y_)))
        
        whoami = [x for x in self.__class__.__bases__ if (x != CustomModel and x != CalibratedClassifierModel)][0]
        
        
        kwargs_classification = copy.deepcopy(self.params_base)
        kwargs_update = dict(num_classes=2, objective='binary:logistic', eval_metric='logloss', labels=[0, 1],
                             score_f_name='LOGLOSS')
        kwargs_classification.update(kwargs_update)
        kwargs_classification.pop('base_score', None)
        for k in kwargs:
            if k in kwargs_classification:
                kwargs[k] = kwargs_classification[k]
        model_classification = whoami(context=self.context,
                                      unfitted_pipeline_path=self.unfitted_pipeline_path,
                                      transformed_features=self.transformed_features,
                                      original_user_cols=self.original_user_cols,
                                      date_format_strings=self.date_format_strings, **kwargs_classification)
        eval_set_classification = None
        if eval_set is not None:
            eval_set_y = self.le.transform(eval_set[0][1])
            eval_set_classification = [(eval_set[0][0], eval_set_y.astype(int))]
        
        if sample_weight is not None:
            sample_weight_ = sample_weight[tr_indx]
        else:
            sample_weight_ = sample_weight

        model_classification.fit(X[tr_indx,:], y_.astype(int)[tr_indx],
                                 sample_weight=sample_weight_, eval_set=eval_set_classification,
                                 sample_weight_eval_set=sample_weight_eval_set, **kwargs)
        
        #calibration
        model_classification.predict_proba = model_classification.predict_simple
        model_classification.classes_ = np.unique(y_)
        calibrator = CalibratedClassifierCV(
            base_estimator=model_classification, 
            method=self.params["calib_method"], 
            cv='prefit')
        calibrator.fit(X[te_indx,:].to_pandas(), y_.astype(int)[te_indx].ravel())
        #calibration
        
        varimp = model_classification.imp_features(columns=X.names)
        varimp.index = varimp['LInteraction']
        varimp = varimp['LGain']
        varimp = varimp[:len(X.names)]
        varimp = varimp.reindex(X.names).values
        importances = varimp
        
        iters = model_classification.best_iterations
        iters = int(max(1, iters))
        self.set_model_properties(model={ 
            "models": calibrator,
            "classes_": self.le.classes_, 
            "best_iters": iters,
            },
            features=list(X.names), importances=importances, iterations=iters
        )
    
    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        data, _, _, _ = self.get_model_properties()
        model, classes_, best_iters = data["models"], data["classes_"], data["best_iters"]
        
        model.base_estimator.best_iterations = best_iters
        preds = model.predict_proba(X)
        
        return preds


class CalibratedClassifierLGBMModel(CalibratedClassifierModel, LightGBMModel, CustomModel):
    @property
    def has_pred_contribs(self):
        return False

    @property
    def has_output_margin(self):
        return False
    
    def set_default_params(self, **kwargs):
        super().set_default_params(**kwargs)
        self.params["calib_method"] = "sigmoid"
        self.params["calib_perc"] = .1
        
    def mutate_params(self, **kwargs):
        super().mutate_params(**kwargs)
        self.params["calib_method"] = np.random.choice(["isotonic", "sigmoid"])
        self.params["calib_perc"] = np.random.choice([.05,.1,.15,.2])
