"""Text classification model using TFIDF"""
import random
import numpy as np
import scipy as sp
import datatable as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from h2oaicore.models import CustomModel
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config, remove, user_dir, arch_type, get_num_gpus_for_prediction
from sklearn.externals import joblib
import copy
import ast
import scipy as sc
import pandas as pd


def get_value(config, key):
    if  key in config.recipe_dict:
        return config.recipe_dict[key]
    elif "config_overrides" in config.get_overrides_dict():
        data = config.get_overrides_dict()["config_overrides"]
        data = ast.literal_eval(ast.literal_eval(data))
        return data.get(key, None)
    else:
        return None
    
    

# Text column should be passed through this transformer for the TextTFIDF model
class TextIdentityTransformer(CustomTransformer):
    """Identity transformer for text"""
    _numeric_output = False

    @property
    def display_name(self):
        return "Str"

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str)

    
#"{'Custom_TextTFIDF_save':'/home/dmitry/Desktop/tmp/save_0.pkl'}"
#"{'Custom_TextTFIDF_load':'/home/dmitry/Desktop/tmp/save_0.pkl','Custom_TextTFIDF_save':'/home/dmitry/Desktop/tmp/save_1.pkl'}"

class TextTFIDFModel(CustomModel):
    """Text classification / regression model using TFIDF"""
    _regression = False
    _binary = True
    _multiclass = True
    _can_handle_non_numeric = True
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _included_transformers = ["TextIdentityTransformer"]  # Takes input only from above transformer
    
    load_key = "Custom_TextTFIDF_load"
    save_key = "Custom_TextTFIDF_save"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_path = None
        self.prev_params = None
        
    @staticmethod
    def reverse_sigmoid(x):
        return np.log(x/(1-x))
    
    @staticmethod
    def inverse_idf(idf_, N_):
        tmp = np.exp(idf_ - 1)
        tmp = np.round((N_+1) / tmp) - 1
        return tmp
    
    def return_tfidf_params(self):
        params_ = {}
        for k in ["max_features", "ngram_range", "norm", "max_df", "min_df"]:
            params_[k] = self.params[k]
        return params_
    
    
    def sync_vectorizers(self, old, new):
        #sync old and new versions
        freq2 = self.inverse_idf(new.idf_, new.N_)
        freq = self.inverse_idf(old.idf_, old.N_)

        #adjust vocabulary and stop word list based on newly data
        #adjust frequency terms and idf terms
        new_freq = []
        remapped_freq = np.zeros(len(freq))
        dict_ = copy.copy(old.vocabulary_)
        stop_list = copy.copy(old.stop_words_)
        max_val = len(dict_)

        for k in new.vocabulary_:
            val = dict_.get(k, -1)
            if val == -1:
                dict_[k] = max_val
                existed = stop_list.discard(k)
                max_val+=1
                new_freq.append(freq2[new.vocabulary_[k]])
            else:
                remapped_freq[val] = freq2[new.vocabulary_[k]]

        old.vocabulary_ = dict_
        old.stop_words_ = stop_list

        freq = freq + remapped_freq
        freq = np.hstack([freq, new_freq])

        old.N_ = old.N_ + new.N_
        freq = np.log((old.N_+1) / (1+freq)) + 1
        old.idf_ = freq
        return old
    
    def sync_tfidf(self,old, new):
        newCols = new.shape[1] - old.shape[1]
        if newCols > 0:
            newCols = np.zeros((old.shape[0], newCols))
            new_tf_idf = sc.sparse.hstack([old, newCols])
        else:
            new_tf_idf = old
        XX = sc.sparse.vstack([new_tf_idf, new])
        return XX
    
    def return_lin_params(self):
        params_ = {}
        for k in ["random_state", "penalty", "C"]:
            params_[k] = self.params[k]
        return params_
    
    def return_rf_params(self):
        params_ = {}
        for k in ["n_estimators", "num_leaves", "reg_alpha", "reg_lambda"]:
            params_[k] = self.params[k]
        return params_
    
    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        if self.load_path and self.prev_params:
            self.params = self.prev_params
        else:
            self.params = dict(
                max_features=kwargs.get("max_features", None),
                ngram_range=kwargs.get("ngram_range", (1, 1)),
                norm = None,
                random_state=2019,
                max_df = 0.9,
                min_df = 3,
                penalty = kwargs.get("penalty", "l2"),
                C = kwargs.get("C", 1.),
                add_rf = kwargs.get("add_rf", False),
                rf_alpha = kwargs.get("rf_alpha", .5),
                n_estimators=kwargs.get("n_estimators", 100),
                num_leaves = kwargs.get("num_leaves", 128),
                reg_alpha = kwargs.get("reg_alpha", .1),
                reg_lambda = kwargs.get("reg_lambda", .1),
            )

    def mutate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params["max_features"] = None#np.random.choice([50000, 100000, None])
        self.params["ngram_range"] = random.choice([(1, 1), (1, 2), (1, 3)])
        self.params["max_df"] = 0.9
        self.params["min_df"] = 3
        self.params["norm"] = None
        self.params["random_state"] =2019
        self.params["penalty"] = random.choice(["l2", "l1"])
        self.params["C"] = random.choice([1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3])
        self.params["add_rf"] = random.choice([False, True])
        self.params["rf_alpha"] = random.choice([.1,.2,.3,.4,.5,.6,.7,.8,.9])
        self.params["n_estimators"] = random.choice([10,20,50,100])
        self.params["num_leaves"] = random.choice([4,16,32,64,128,256])
        self.params["reg_alpha"] = random.choice([0,.1,.5,1.,2.,10])
        self.params["reg_lambda"] = random.choice([0,.1,.5,1.,2.,10])
        
    def _fit_vectorizer(self, vec, data):
        try_flag = True
        min_df = self.params["min_df"]
        max_df = self.params["max_df"]
        while try_flag:
            try:
                vec.set_params(
                    min_df=min_df,
                    max_df=max_df,
                )
                vec.fit(data) 
                try_flag = False
            except ValueError:
                if min_df == 1 and max_df == 1.0:  # if min_df is 1 & max_df == 1.0 no more tries left
                    try_flag = False
                    raise
                min_df -= 1
                max_df += 0.05
                max_df = min(max_df, 1.0)  # limiting max_df to 1.0
                min_df = max(min_df, 1)  # limiting min_df to 1
                try_flag = True
                
        self.params["min_df"] = min_df
        self.params["max_df"] = max_df
        return vec

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        y_ = y.copy()
        orig_cols = list(X.names)

        self.loaded = False

        self.load_path = get_value(config, self.load_key)
        self.save_path = get_value(config, self.save_key)

        if self.load_path:
            data = joblib.load(self.load_path)
            self.tfidf_objs = data["tf_idf_obj"]
            self.tf_idf_data = data["tf_idf_data"]
            self.prev_params = data["params"]
            self.target = data["target"]
            self.loaded = True

        if not self.loaded:
            if self.num_classes >= 2:
                lb = LabelEncoder()
                lb.fit(self.labels)
                y = lb.transform(y)

            self.tfidf_objs = {}
            self.tf_idf_data = {}
            new_X = None
            for col in X.names:
                XX = X[:, col].to_pandas()
                XX = XX[col].astype(str).fillna("NA").values.tolist()
                tfidf_vec = TfidfVectorizer(**self.return_tfidf_params())
                tfidf_vec = self._fit_vectorizer(tfidf_vec, XX)
                XX = tfidf_vec.transform(XX)
                tfidf_vec.N_ = XX.shape[0]
                self.tfidf_objs[col] = tfidf_vec
                self.tf_idf_data[col] = XX
                if new_X is None:
                    new_X = XX
                else:
                    new_X = sp.sparse.hstack([new_X, XX])
        else:
            y_ = np.hstack([self.target, y_])
            y = y_.copy()
            if self.num_classes >= 2:
                lb = LabelEncoder()
                lb.fit(self.labels)
                y = lb.transform(y)
            
            new_X = None
            for col in X.names:
                XX = X[:, col].to_pandas()
                XX = XX[col].astype(str).fillna("NA").values.tolist()
                N_ = len(XX)
                tfidf_vec = TfidfVectorizer()
                tfidf_vec.set_params(**self.tfidf_objs[col].get_params())
                try:
                    tfidf_vec.fit(XX)
                    new_data_avail = True
                except ValueError:
                    new_data_avail = False
                if new_data_avail:
                    tfidf_vec.N_ = N_
                    pre_trained = self.tfidf_objs[col]
                    pre_trained = self.sync_vectorizers(pre_trained, tfidf_vec)
                else:
                    pre_trained = self.tfidf_objs[col]
                
                XX = pre_trained.transform(XX)
                self.tfidf_objs[col] = pre_trained
                
                XX = self.sync_tfidf(self.tf_idf_data[col], XX)
                self.tf_idf_data[col] = XX
                if new_X is None:
                    new_X = XX
                else:
                    new_X = sp.sparse.hstack([new_X, XX])
        
        models = [LogisticRegression(**self.return_lin_params())]
        if self.params["add_rf"]:
            from h2oaicore.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb
            import lightgbm as lgbm
            models.append(lgbm.LGBMClassifier(
                boosting_type='rf',
                colsample_bytree=.5,
                subsample=.632, # Standard RF bagging fraction
                min_child_weight=2.5,
                min_child_samples=5,
                subsample_freq=1,
                min_split_gain=0,
                n_jobs=-1,
                **self.return_rf_params()
            ))
        
        for m in models:
            m.fit(new_X, y)
        
        importances = [1] * len(orig_cols)
        self.set_model_properties(
            model={
                "model": models,
                "tf-idfs": self.tfidf_objs
            },
            features=orig_cols,
            importances=importances,
            iterations=0
        )
        if self.save_path:
            joblib.dump({
                "tf_idf_obj": self.tfidf_objs,
                "tf_idf_data": self.tf_idf_data,
                "params": self.params,
                "target": y_,
                },
                self.save_path
            )
        # clear large objects to avoid large data in subprocess pipe
        self.tfidf_objs = None
        self.tf_idf_data = None

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        new_X = None
        data, _, _, _ = self.get_model_properties()
        models = data["model"]
        self.tfidf_objs = data["tf-idfs"]
        for col in X.names:
            XX = X[:, col].to_pandas()
            XX = XX[col].astype(str).fillna("NA").values.tolist()
            tfidf_vec = self.tfidf_objs[col]
            XX = tfidf_vec.transform(XX)
            if new_X is None:
                new_X = XX
            else:
                new_X = sp.sparse.hstack([new_X, XX])
        preds = []
        for m in models:
            if self.num_classes == 1:
                preds.append(m.predict(new_X).reshape(-1,1))
            else:
                preds.append(m.predict_proba(new_X))
        if len(preds)> 1:
            preds = np.dstack(preds[:2])
            preds = np.average(preds, axis = 2, weights = [(1.-self.params["rf_alpha"]), self.params["rf_alpha"]])
        else:
            preds = preds[0]
        return preds

    def pre_get_model(self):  # copy-paste from LightGBM model class
        from h2oaicore.lightgbm_dynamic import got_cpu_lgb, got_gpu_lgb

        if arch_type == 'ppc64le':
            # ppc has issues with this, so force ppc to only keep same architecture
            return
        if self.self_model_was_not_set_for_predict:
            # if no self.model, then should also not have imported lgbm/xgb yet
            # well, for rulefit not true
            #check_no_xgboost_or_lightgbm()
            pass
        # Force the C++ session to have good "defaults" by making an exemplary dummy model to set the state from which
        # our model will later inherit default settings from.
        try:
            import lightgbm as lgb
            params = dict(n_jobs=self.params_base['n_jobs'])
            if get_num_gpus_for_prediction() == 0 or arch_type == 'ppc64le':  # power has no GPU for lgbm yet
                params['device_type'] = 'cpu'
            else:
                params['device_type'] = 'gpu'
                params['gpu_device_id'] = self.params_base.get('gpu_id', 0)
                params['gpu_platform_id'] = 0
                params['gpu_use_dp'] = config.reproducible

            model = lgb.LGBMClassifier(**params)
            X = np.array([[1, 2, 3, 4], [1, 3, 4, 2]])
            y = np.array([1, 0])
            model.fit(X=X, y=y)
        except:
            if config.hard_asserts:
                raise

