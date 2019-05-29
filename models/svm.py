import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from h2oaicore.models import CustomModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge # will be used to derive feature importances
from sklearn.svm import SVC, SVR


class LibSvmModel(CustomModel):
    
    _regression = True
    _binary = True
    _multiclass = True  
    
    _boosters = ['svm']
    _display_name = "SVMModel"
    _description = "Support Vector Machine model for regression and classification using the libsvm method of sklearn . Not adviced if the data is larger than 100K rows. In that case a linear implementation should be preferred isntead."


    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        
        C = max(kwargs['C'],0.00001) if 'C' in kwargs else 1.
        epsilon = max(kwargs['epsilon'],0.00001) if 'epsilon' in kwargs else 0.1       
        kernel =kwargs['kernel'] if "kernel" in  kwargs  and kwargs['kernel'] in ["linear","poly","rbf","sigmoid"] else "rbf"
        
        self.params = {'C': C,
                       'kernel': kernel,
                       'epsilon': epsilon ,
                       }

    def mutate_params(self,
                      accuracy, time_tolerance, interpretability,
                      **kwargs):
        
        list_of_C=[0.001,0.01,0.1,1.,2.5,5.,10.]
        list_of_kernel=["linear","poly","rbf","sigmoid"]
        list_of_epsilon=[0.001,0.01,0.1,1.,2.5,5.,10.]
        
                             
        C_index=np.random.randint(0, high=len(list_of_C)) 
        kernel_index=np.random.randint(0, high=len(list_of_kernel)) 
        epsilon_index=np.random.randint(0, high=len(list_of_epsilon))         
        
        C=  list_of_C[C_index]  
        kernel=  list_of_kernel[kernel_index]  
        epsilon=  list_of_epsilon[epsilon_index]          
        
        self.params = {'C': C,
                       'kernel': kernel,
                       'epsilon': epsilon ,
                       }            



    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)

        orig_cols = list(X.names)
        feature_model=Ridge(alpha=1., random_state=self.random_state)
        
        if self.num_classes >= 2:

            model = SVC(C=self.params["C"], kernel=self.params["kernel"], probability=True, random_state=self.random_state)      
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
        else:
            model = SVR(C=self.params["C"], kernel=self.params["kernel"], epsilon=self.params["epsilon"])
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
        if self.num_classes >= 2: 
             feature_model.fit(X, y, sample_weight=sample_weight)
             model.fit(X, y, sample_weight=sample_weight)
        else :
             feature_model.fit(X, y)
             model.fit(X, y)         
             
        importances=np.array(abs(feature_model.coef_))
            
        
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
            raise NotImplementedError("No Shapley for K-nearest model")
