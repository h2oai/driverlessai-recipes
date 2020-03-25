"""Extract sentiment from text using pretrained models from TextBlob"""
import importlib
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.transformers import TextTransformer
import datatable as dt
import numpy as np
from h2oaicore.systemutils import config, remove, temporary_files_path
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import Pipeline
import ast
import copy

def get_value(config, key):
    if  key in config.recipe_dict:
        return config.recipe_dict[k]
    elif "config_overrides" in config.get_overrides_dict():
        data = config.get_overrides_dict()["config_overrides"]
        data = ast.literal_eval(ast.literal_eval(data))
        return data.get(key, None)
    else:
        return None

# """   
# {
# 'Custom_TextTransformer_load':'/home/dmitry/Desktop/tmp/saved_txtTransformerZZZZ.pkl',
# 'Custom_TextTransformer_save':'/home/dmitry/Desktop/tmp/saved_txtTransformerYYYY.pkl'
# }
# """
class Cached_TextTransformer(CustomTransformer):
    _regression = True
    _binary = True
    _multiclass = True
    
    _display_name = "Cached_TextTransformer"
    load_key = "Custom_TextTransformer_load"
    save_key = "Custom_TextTransformer_save"
    
    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_parameter_choices():
        return {
            "max_features": [None],
            "tf_idf": [True, False],
            "max_ngram": [1, 2, 3],
            "dim_reduction": [50],
        }
    
    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def __init__(self, max_features = None, tf_idf = True, max_ngram = 1, dim_reduction = 50, **kwargs):
        super().__init__(**kwargs)
        
        self.loaded = False

        self.load_path = get_value(config, self.load_key)
        self.save_path = get_value(config, self.save_key)
        
        if not self.load_path:
            self.TextTransformer = TextTransformer(
                max_features = max_features, 
                tf_idf=tf_idf, 
                max_ngram = max_ngram, 
                dim_reduction = dim_reduction,
                **kwargs
            )
        else:
            self.TextTransformer = joblib.load(self.load_path) 
            self.loaded = True
    

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.TextTransformer.N_ = X.shape[0]
        result = self.TextTransformer.fit_transform(X.to_pandas())
        
        if self.save_path:
            joblib.dump(self.TextTransformer, self.save_path)
        return result

    def transform(self, X: dt.Frame):
        return self.TextTransformer.transform(X.to_pandas())
    
    _mojo = True
    from h2oaicore.mojo import MojoWriter, MojoFrame

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):
        return self.TextTransformer.write_to_mojo(mojo, iframe, group_uuid, group_name)


class Updatable_TextTransformer_TFIDFOnly(Cached_TextTransformer):
    """
    Only updates TF-IDF terms, vocabulary and stop word list remain the same
    """
    _display_name = "Updatable_TextTransformer_TFIDFOnly"
    
    @staticmethod
    def inverse_idf(idf_, N_):
        tmp = np.exp(idf_ - 1)
        tmp = np.round((N_+1) / tmp) - 1
        return tmp
    
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        if self.loaded:
            X_ = X.to_pandas()
            N_ = len(X_)
            for col in self.input_feature_names:
                if self.TextTransformer.tf_idf: # update tf-idf terms for tokens in new data
                    cv = TfidfVectorizer()
                    pre_trained = self.TextTransformer.pipes[col][0]["model"]
                    cv.set_params(**pre_trained.get_params())
                    cv.set_params(**{
                        "vocabulary": pre_trained.vocabulary_, 
                        "stop_words": pre_trained.stop_words_
                    })
                    pipe_ = copy.deepcopy(self.TextTransformer.pipes[col][0])
                    new_pipe = []
                    for step in pipe_.steps:
                        if step[0] != 'model':
                            new_pipe.append(step)
                        else:
                            new_pipe.append(('model', cv))
                            break
                    new_pipe = Pipeline(new_pipe)
                    new_pipe.fit(self.TextTransformer.stringify_col(X_[col]))
                    
                    freq2 = self.inverse_idf(cv.idf_, N_)
                    
                    freq = inverse_idf(
                        pre_trained.idf_, 
                        pre_trained.N_
                    )
                    freq = freq + freq2
                    pre_trained.N_ = pre_trained.N_ + N_
                    freq = np.log((pre_trained.N_+1) / (1+freq)) + 1
                    pre_trained.idf_ = freq
            
            result = self.TextTransformer.transform(X.to_pandas())

        else:
            self.TextTransformer.N_ = X.shape[0]
            result = self.TextTransformer.fit_transform(X.to_pandas())
        
        if self.save_path:
            joblib.dump(self.TextTransformer, self.save_path)
        return result

    

