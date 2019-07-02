"""LibFM implementation of fastFM """
import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from h2oaicore.models import CustomModel
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
#paper: https://arxiv.org/abs/1505.00641


class fastFMModel(CustomModel):
    
    _regression = True
    _binary = True
    _multiclass = False  # WIP
    _display_name = "fastFM"
    _description = "LibFM implementation of fastFM"
    # als.FMRegression(n_iter=1000, init_stdev=0.1, rank=2, l2_reg_w=0.1, l2_reg_V=0.5)

    _modules_needed_by_name = ['fastFM']
   
    
    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):

        n_iter = min(max(kwargs['n_iter'], 1),100000) if 'n_iter' in kwargs else 100                    
        init_stdev = max(kwargs['init_stdev'], 0.000001) if 'init_stdev' in kwargs else 0.1                        
        rank = min(max(kwargs['rank'], 0),2) if 'rank' in kwargs else 2    
        l2_reg_w = max(kwargs['l2_reg_w'], 0.000000001) if 'l2_reg_w' in kwargs else 0.1                   
        l2_reg_V = max(kwargs['l2_reg_V'], 0.000000001) if 'l2_reg_V' in kwargs else 0.5     
        #random_state = kwargs['random_state'] if 'random_state' in kwargs else 12345                         

        self.params = {'n_iter': n_iter,
                       'init_stdev': init_stdev,
                       'rank': rank,
                       'l2_reg_w': l2_reg_w,
                       'l2_reg_V': l2_reg_V,
                       }

    def mutate_params(self,
                      accuracy, time_tolerance, interpretability,
                      **kwargs):

        
        if accuracy > 8:
            list_of_n_iter = [200, 300, 400, 500, 1000, 2000]
        elif accuracy >= 5:
            list_of_n_iter = [50, 100, 200, 300, 400, 500]
        else:
            list_of_n_iter = [10, 50, 100, 150, 200, 250, 300]        
        
        
        
        list_of_init_stdev = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.]    
        list_of_reg_w = [0.0001, 0.001, 0.01, 0.1, 1., 3., 10.]
        list_of_l2_reg_V = [0.001, 0.01, 0.1, 1., 3., 10., 20.]

        n_iter_index = np.random.randint(0, high=len(list_of_n_iter))
        reg_w_index = np.random.randint(0, high=len(list_of_reg_w))
        reg_V_index = np.random.randint(0, high=len(list_of_l2_reg_V))
        init_stdev_index = np.random.randint(0, high=len(list_of_init_stdev))
        
        
        n_iter = list_of_n_iter[n_iter_index]
        reg_w = list_of_reg_w[reg_w_index]
        reg_V = list_of_l2_reg_V[reg_V_index]
        init_stdev = list_of_init_stdev[init_stdev_index]                
        rank=2
        

        self.params = {'n_iter': n_iter,
                       'init_stdev': init_stdev,
                       'rank': rank,
                       'l2_reg_w': reg_w,
                       'l2_reg_V': reg_V,
                       }

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        
        from fastFM import als
        X = dt.Frame(X)

        orig_cols = list(X.names)

        if self.num_classes >= 2:
            model = als.FMClassification(n_iter=self.params["n_iter"], init_stdev=self.params["init_stdev"],
                                       rank=self.params["rank"], l2_reg_w=self.params["l2_reg_w"],
                                                       l2_reg_V=self.params["l2_reg_V"], random_state=self.random_state)
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            y[y!=1]=-1
             
        else:
            model = als.FMRegression(n_iter=self.params["n_iter"], init_stdev=self.params["init_stdev"],
                                       rank=self.params["rank"], l2_reg_w=self.params["l2_reg_w"],
                                                       l2_reg_V=self.params["l2_reg_V"], random_state=self.random_state)

        self.means = dict()
        self.scaler = StandardScaler()
        for col in X.names:
            XX = X[:, col]
            self.means[col] = XX.mean1()
            if np.isnan(self.means[col]):
                self.means[col] = 0
            XX.replace(None, self.means[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        X = X.to_numpy()
        X = self.scaler.fit_transform(X)
        X=csr_matrix(X) #requires sparse matrix        
        model.fit(X, y)
        importances= np.array(abs(model.w_)) 
        

        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),  # abs(model.coef_[0])
                                  iterations=0)

    def predict(self, X, **kwargs):
        from fastFM import als
        X = dt.Frame(X)
        for col in X.names:
            XX = X[:, col]
            XX.replace(None, self.means[col])
            X[:, col] = XX

        pred_contribs = kwargs.get('pred_contribs', None)
        output_margin = kwargs.get('output_margin', None)

        model, _, _, _ = self.get_model_properties()
        X = X.to_numpy()
        X = self.scaler.transform(X)
        X=csr_matrix(X) #requires sparse matrix
        if not pred_contribs:
            if self.num_classes == 1:
                preds = model.predict(X)
            else:
                preds = np.array(model.predict_proba(X))
                preds = np.column_stack((1-preds, preds))
                # preds = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            return preds
        else:
            raise NotImplementedError("No Shapley for SVM")