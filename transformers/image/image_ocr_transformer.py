"""Extracts text from images using OCR"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np

class ImageOCRTextTransformer(CustomTransformer):

    _modules_needed_by_name = ['Pillow==5.0.0',"pytesseract==0.3.0"]  
    _parallel_task = True  # if enabled, params_base['n_jobs'] will be >= 1 (adaptive to system), otherwise 1
    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _can_use_multi_gpu = True  # if enabled, can get access to multiple GPUs for single transformer (experimental)

    
    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)
   
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)
 
    def transform(self, X: dt.Frame):
        
        import pytesseract
        from PIL import Image

        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: pytesseract.image_to_string(Image.open(x)))
