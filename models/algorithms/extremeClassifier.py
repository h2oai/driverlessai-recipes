""" Extreme Classifier Model: To speed up train of multiclass model (100s of classes) for lightGBM.
    Caution: can only be used for AUC (or GINI) and accuracy metrics.
    Based on: Extreme Classification in Log Memory using Count-Min Sketch: https://arxiv.org/abs/1910.13830
"""

import copy
import datatable as dt
from h2oaicore.systemutils import config
from h2oaicore.models import CustomModel, LightGBMModel
from sklearn.preprocessing import LabelEncoder
import numpy as np
import hashlib
from scipy.special import softmax

class HashLabel():
    def __init__(self, sigma = .75, B = 2, random_state = 1):
        self.n_class = None
        self.sigma = sigma
        self.B = B
        self.random_state = random_state
        
        self.le = LabelEncoder()
        
    def fit(self, labels):
        classes = np.unique(labels.astype(int))
        
        self.n_class = int(len(classes))
        classes_ = self.le.fit_transform(classes)#classes.astype(int)
        
        self.R = int((2*np.log(self.n_class/np.sqrt(self.sigma)))/np.log(self.B))
        
        maping_created = False
        np.random.seed(self.random_state)
        while not(maping_created):
            self.mapped_classes = np.zeros((self.n_class, self.R), dtype = int)
            for c in classes_:
                md5 = hashlib.md5(str(hash(c)).encode('utf-8'))
                for r in range(self.R):
                    md5.update(str(np.random.randint(np.iinfo(np.int16).max)).encode('utf-8'))
                    self.mapped_classes[c,r] = int(md5.hexdigest(), 16) % self.B
        
            unique_rows = np.vstack({tuple(row) for row in self.mapped_classes})
            maping_created = unique_rows.shape != self.mapped_classes.shape
        
    def transform(self, labels):
        labels_ = self.le.transform(labels.ravel().astype(int))

        labels_set = np.zeros((labels_.shape[0], self.mapped_classes.shape[1]), dtype = int)

        for r in range(self.mapped_classes.shape[1]):
            labels_set[:,r] = self.mapped_classes[labels_,r]
        return labels_set
    
    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)
    
    def predict(self, probas):
        resulting_preds = np.zeros((np.max([len(x) for x in probas]),self.n_class))
        for i,c in enumerate(self.mapped_classes):
            for indx, r in enumerate(c):
                resulting_preds[:,i] += probas[indx][:,r]
        resulting_preds = resulting_preds / self.R
        #return resulting_preds/resulting_preds.sum(axis = 1).reshape(-1,1)
        return softmax(resulting_preds, axis=1)
    
class ExtremeClassifierModel:
    _regression = False
    _binary = False
    _multiclass = True
    _can_use_gpu = True
    _mojo = False
    _description = "Extreme Classifier Model"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    le = LabelEncoder()

    @staticmethod
    def do_acceptance_test():
        """
        Return whether to do acceptance tests during upload of recipe and during start of Driverless AI.
        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return False

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        assert len(self.__class__.__bases__) == 3
        assert ExtremeClassifierModel in self.__class__.__bases__
        
        self.le.fit(self.labels)
        y_ = self.le.transform(y)
        
        whoami = [x for x in self.__class__.__bases__ if (x != CustomModel and x != ExtremeClassifierModel)][0]

        hashL = HashLabel(
            B = 2, 
            sigma = .75 if len(self.le.classes_) < 30 else 1 / len(self.le.classes_), 
            random_state = 43155
        )
        y_hashed = hashL.fit_transform(y_)
        
        # Models - Binary classifiers trained on all samples on binned orignal classes
        models = []
        for c in range(y_hashed.shape[1]):
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
                eval_set_y = hashL.transform(eval_set_y)
                eval_set_classification = [(eval_set[0][0], eval_set_y[:,c].astype(int))]
                
                
            model_classification.fit(X, y_hashed[:,c].astype(int),
                                     sample_weight=sample_weight, eval_set=eval_set_classification,
                                     sample_weight_eval_set=sample_weight_eval_set, **kwargs)
            models.append(model_classification)

        importances = None
        for varimp in [x.imp_features(columns=X.names) for x in models]:
            # get normalized varimp vector in order of X.names, so can add between models
            varimp.index = varimp['LInteraction']
            varimp = varimp['LGain']
            varimp = varimp[:len(X.names)]
            varimp = varimp.reindex(X.names).values
            if importances is None:
                importances = varimp
            else:
                importances += varimp
        importances /= len(models)
        iters = np.mean([x.best_iterations for x in models])
        iters = int(max(1, iters))
        self.set_model_properties(model={
            "models":models, 
            "classes_": self.le.classes_, 
            "y_hash": hashL,
            "best_iters": [x.best_iterations for x in models],
            },
            features=list(X.names), importances=importances, iterations=iters
        )
    
    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        data, _, _, _ = self.get_model_properties()
        models, classes_, hashL, best_iters = data["models"], data["classes_"], data["y_hash"], data["best_iters"]
        
        preds_ = []
        for m, best_i in zip(models, best_iters):
            m.best_iterations = best_i
            preds = m.predict(X, **kwargs)
            preds_.append(preds)
        preds = hashL.predict(preds_)
        
        return preds


class ExtremeClassifierLGBMModel(ExtremeClassifierModel, LightGBMModel, CustomModel):
    @property
    def has_pred_contribs(self):
        return False

    @property
    def has_output_margin(self):
        return False
