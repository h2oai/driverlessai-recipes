import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from h2oaicore.models import CustomModel
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class linsvc(BaseEstimator, ClassifierMixin):
  def __init__(self,
            random_state=1
      ):    
    self.model=LinearSVC(penalty="l2", loss="squared_hinge", C=1.0,random_state=random_state)
    self.random_state=random_state
    self.classes_=[0,1]

  def fit(self, X, y, sample_weight=None):
    self.model.fit(X, y, sample_weight=sample_weight)
      
    return self

  def predict(self, X):  #this predicts classification

    preds = self.model.predict(X )  
    return preds
  
  def predict_proba(self, X): 
    X1=X.dot(self.model.coef_[0])
    return np.column_stack((np.array(X1)-1,np.array(X1) ))   

  def set_params(self,random_state=1):
        self.model. set_params(random_state=random_state)    
    
  def get_params(self, deep=False):
      return  {"random_state":self.random_state}

  def get_coeff(self):
      return  self.model.coef_[0]
  

class LinearSVMModel(CustomModel):
    
    _regression = True
    _binary = True
    _multiclass = False  # WIP
    
    _boosters = ['linearsvm']
    _display_name = "LinearSVM"
    _description = "Linear Support Vector Machine with the Liblinear method + Calibration for probabilities"

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        orig_cols = list(X.names)

        if self.num_classes >= 2:
            mod=linsvc(random_state=self.random_state)
            kf=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            model = CalibratedClassifierCV(base_estimator=mod, method='isotonic',cv=kf)            
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
        else:
            model = LinearSVR(epsilon=0.0, C=1.0, loss="epsilon_insensitive",  random_state=self.random_state)
        self.means = dict()
        self.scaler=StandardScaler()
        for col in X.names:
            XX = X[:, col]
            self.means[col] = XX.mean1()
            if np.isnan(self.means[col]):
                self.means[col] = 0
            XX.replace(None, self.means[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()
        X=self.scaler.fit_transform(X)
        model.fit(X, y, sample_weight=sample_weight)
        if self.num_classes >= 2:
            importances=np.array([0.0 for k in range (len(orig_cols))])
            for classifier in model.calibrated_classifiers_ :
                importances+=np.array(abs(classifier.base_estimator.get_coeff()))
        else :
            importances=np.array(abs(model.coef_[0]))
            
        
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),#abs(model.coef_[0])
                                  iterations=0)

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        for col in X.names:
            XX = X[:, col]
            XX.replace(None, self.means[col])
            X[:, col] = XX

        pred_contribs = kwargs.get('pred_contribs', None)
        output_margin = kwargs.get('output_margin', None)

        model, _, _, _ = self.get_model_properties()
        X=X.to_numpy()
        X=self.scaler.transform(X)
        if not pred_contribs:
            if self.num_classes == 1:
                preds = model.predict(X)
            else:
                preds = model.predict_proba(X)
                #preds = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            return preds
        else:
            raise NotImplementedError("No Shapley for SVM")
