"""Extract the count of nouns, verbs, adjectives and adverbs in the text"""
import datatable as dt
import numpy as np
import shutil
import os
from zipfile import ZipFile
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config, remove, user_dir
from h2oaicore.systemutils_more import download


class POSTagTransformer:
    """Transformer to extract the count of POS tags"""
    _method = NotImplemented
    _modules_needed_by_name = ["nltk==3.4.3"]
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import nltk
        nltk_data_path = os.path.join(user_dir(), config.contrib_env_relative_directory, "nltk_data")
        nltk_temp_path = os.path.join(user_dir(), "nltk_data")
        nltk.data.path.append(nltk_data_path)
        try:
            self.pos_tagger = nltk.pos_tag
            self.pos_tagger("test")
        except LookupError:
            os.makedirs(nltk_data_path, exist_ok=True)
            os.makedirs(nltk_temp_path, exist_ok=True)
            tagger_path = os.path.join(nltk_data_path, "taggers")
            os.makedirs(tagger_path, exist_ok=True)
            file1 = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip",
                             dest_path=nltk_temp_path)
            file2 = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/maxent_treebank_pos_tagger.zip",
                             dest_path=nltk_temp_path)
            self.unzip_file(file1, tagger_path)
            self.unzip_file(file2, tagger_path)
            self.atomic_copy(file1, tagger_path)
            self.atomic_copy(file2, tagger_path)
            self.pos_tagger = nltk.pos_tag
            self.pos_tagger("test")

    def unzip_file(self, src, dst_dir):
        with ZipFile(src, 'r') as zip_ref:
            zip_ref.extractall(dst_dir)

    def atomic_move(self, src, dst):
        try:
            shutil.move(src, dst)
        except shutil.Error:
            pass
        remove(src)

    def atomic_copy(self, src=None, dst=None):
        import uuid
        my_uuid = uuid.uuid4()
        src_tmp = src + str(my_uuid)
        shutil.copy(src, src_tmp)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        self.atomic_move(src_tmp, dst)
        remove(src_tmp)

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def get_pos_count(self, text):
        pos_tag = self.__class__._method
        pos_tagged_text = self.pos_tagger(text.split())
        return len([word for word, pos in pos_tagged_text if pos[0] == pos_tag])

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).fillna("NA").iloc[:, 0].apply(lambda x: self.get_pos_count(x))


class NounCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of nouns in the text column"""
    _method = "N"


class VerbCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of verbs in the text column"""
    _method = "V"


class AdjectiveCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of adjectives in the text column"""
    _method = "J"


class AdverbCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of adverbs in the text column"""
    _method = "R"
