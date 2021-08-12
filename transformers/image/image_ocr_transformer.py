"""Convert a path to an image to text using OCR based on tesseract"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class ImageOCRTextTransformer(CustomTransformer):
    _modules_needed_by_name = ['pillow==8.3.1', "pytesseract==0.3.0"]
    _parallel_task = True  # if enabled, params_base['n_jobs'] will be >= 1 (adaptive to system), otherwise 1
    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _can_use_multi_gpu = True  # if enabled, can get access to multiple GPUs for single transformer (experimental)

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def do_acceptance_test():
        return False

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def image_ocr(self, path):

        import pytesseract
        from PIL import Image

        try:
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
        except:
            text = ''

        return text

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: self.image_ocr(x))
