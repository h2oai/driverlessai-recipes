"""Preprocess the text column by stemming, lemmatization and stop word removal"""
import datatable as dt
import numpy as np
import shutil
import os
from zipfile import ZipFile

import filelock
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config, remove, temporary_files_path
from h2oaicore.systemutils_more import download


class TextPreprocessingTransformer(CustomTransformer):
    """Transformer to preprocess the text"""
    _numeric_output = False
    _is_reproducible = True
    _modules_needed_by_name = ["nltk==3.4.3"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_stemming = True  # turn off as needed
        self.do_lemmatization = True  # turn off as needed
        self.remove_stopwords = True  # turn off as needed

        import nltk
        nltk_data_path = os.path.join(config.data_directory, config.contrib_env_relative_directory, "nltk_data")
        nltk_temp_path = os.path.join(temporary_files_path, "nltk_data")
        nltk.data.path.append(nltk_data_path)
        os.makedirs(nltk_data_path, exist_ok=True)
        nltk_download_lock_file = os.path.join(nltk_data_path, "nltk.lock")
        with filelock.FileLock(nltk_download_lock_file):
            nltk.download('stopwords', download_dir=nltk_data_path)
            nltk.download('punkt', download_dir=nltk_data_path)
            nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)
            nltk.download('maxent_treebank_pos_tagger', download_dir=nltk_data_path)
            nltk.download('wordnet', download_dir=nltk_data_path)
            nltk.download('sonoritysequencing', download_dir=nltk_data_path)

        # download resources for stemming if needed
        if self.do_stemming:
            try:
                self.stemmer = nltk.stem.porter.PorterStemmer()
                self.stemmer.stem("test")
            except LookupError:
                os.makedirs(nltk_data_path, exist_ok=True)
                os.makedirs(nltk_temp_path, exist_ok=True)
                tokenizer_path = os.path.join(nltk_data_path, "tokenizers")
                os.makedirs(tokenizer_path, exist_ok=True)
                file1 = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip",
                                 dest_path=nltk_temp_path)
                self.unzip_file(file1, tokenizer_path)
                self.atomic_copy(file1, tokenizer_path)
                self.stemmer = nltk.stem.porter.PorterStemmer()
                self.stemmer.stem("test")

        # download resources for lemmatization if needed
        if self.do_lemmatization:
            try:
                from nltk.corpus import wordnet
                self.lemmatizer = nltk.stem.WordNetLemmatizer()
                self.pos_tagger = nltk.pos_tag
                self.lemmatizer.lemmatize("test", wordnet.NOUN)
                self.pos_tagger("test")
            except LookupError:
                os.makedirs(nltk_data_path, exist_ok=True)
                os.makedirs(nltk_temp_path, exist_ok=True)
                tagger_path = os.path.join(nltk_data_path, "taggers")
                corpora_path = os.path.join(nltk_data_path, "corpora")
                os.makedirs(tagger_path, exist_ok=True)
                os.makedirs(corpora_path, exist_ok=True)
                file1 = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/averaged_perceptron_tagger.zip",
                                 dest_path=nltk_temp_path)
                file2 = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/taggers/maxent_treebank_pos_tagger.zip",
                                 dest_path=nltk_temp_path)
                file3 = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
                                 dest_path=nltk_temp_path)
                self.unzip_file(file1, tagger_path)
                self.unzip_file(file2, tagger_path)
                self.unzip_file(file3, corpora_path)
                self.atomic_copy(file1, tagger_path)
                self.atomic_copy(file2, tagger_path)
                self.atomic_copy(file3, corpora_path)
                from nltk.corpus import wordnet
                self.lemmatizer = nltk.stem.WordNetLemmatizer()
                self.pos_tagger = nltk.pos_tag
                self.lemmatizer.lemmatize("test", wordnet.NOUN)
                self.pos_tagger("test")
            self.wordnet_map = {"N":wordnet.NOUN,
                                "V":wordnet.VERB,
                                "J":wordnet.ADJ,
                                "R":wordnet.ADV,
                                "O":wordnet.NOUN}

        # download resources for stopwords if needed
        if self.remove_stopwords:
            try:
                self.stopwords = set(nltk.corpus.stopwords.words('english'))
            except LookupError:
                os.makedirs(nltk_data_path, exist_ok=True)
                os.makedirs(nltk_temp_path, exist_ok=True)
                corpora_path = os.path.join(nltk_data_path, "corpora")
                os.makedirs(corpora_path, exist_ok=True)
                file1 = download("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip",
                                 dest_path=nltk_temp_path)
                self.unzip_file(file1, corpora_path)
                self.atomic_copy(file1, corpora_path)
                self.stopwords = set(nltk.corpus.stopwords.words('english'))

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

    @property
    def display_name(self):
        return "PreprocessedText"

    def preprocess(self, text):
        if self.do_stemming:
            text = " ".join([self.stemmer.stem(word) for word in text.split()])
        if self.do_lemmatization:
            pos_tagged_text = self.pos_tagger(text.split())
            text = " ".join([self.lemmatizer.lemmatize(word, self.wordnet_map.get(pos[0], self.wordnet_map["O"]))
                             for word, pos in pos_tagged_text])
        if self.remove_stopwords:
            text = " ".join([word for word in str(text).split()
                             if word.lower() not in self.stopwords])
        return text

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).fillna("NA").iloc[:, 0].apply(lambda x: self.preprocess(x))
