"""Preprocess the tweets by normalising username, removing unnecessatry punctuations, exapanding the hashtags"""
import re
import os
import pandas as pd
import datatable as dt
from h2oaicore.systemutils import config
from h2oaicore.data import CustomData

text_colnames = ["text"]

output_dataset_name = "df_preprocessed"

_global_modules_needed_by_name = ["wordsegment"]
import wordsegment


class PreprocessDataClass(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None):

        if X is None:
            return []
        fixup = process_tweets()

        X = dt.Frame(X).to_pandas()
        for text_colname in text_colnames:
            X["preprocessed_" + text_colname] = X[text_colname].astype(str).apply(
                lambda x: fixup.preprocess(x))

        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)

        # Save files to disk
        file_train = os.path.join(temp_path, output_dataset_name + ".csv")
        X.to_csv(file_train, index=False)

        return [file_train]


class process_tweets:
    """Class for Processing tweets"""

    def __init__(self):
        wordsegment.load()
        self.segment = wordsegment.segment

    @staticmethod
    def currency_replace(text):
        text = re.sub(r"\$", " dollar ", text)
        text = re.sub(r"£", " pound ", text)
        text = re.sub(r"€", " euro ", text)
        text = re.sub(r"¥", " yen ", text)
        text = re.sub(r"[¢₡₱₭₦]", " currency ", text)
        return text

    @staticmethod
    def char_removing(text):
        text = text.replace("http://url.removed", "")
        text = re.sub(r"[ं-ో̇]", "", text)
        text = re.sub(r"[•]", "", text)
        text = re.sub(r"[】【]", "", text)
        text = re.sub(r"[\{\}\(\)\[\]]+", " ", text)
        text = re.sub(r"[*/\&|_<>~\+=\-\^™\\\%]+", " ", text)
        text = re.sub(r"[;:…]+", " ", text)
        return text

    def fix_hashtag(self, text):
        hashtags = re.findall(r"(#\w+)", text)
        for hashtag in hashtags:
            processed_hashtag = '# ' + (' '.join(self.segment(hashtag)))
            text = text.replace(hashtag, processed_hashtag)
        return text

    @staticmethod
    def fix_username(text):
        text = re.sub(r"@[a-zA-Z0-9]+", "@username", text)
        return text

    def preprocess(self, text):
        text = self.currency_replace(text)
        text = self.char_removing(text)
        text = self.fix_hashtag(text)
        text = self.fix_username(text)
        text = re.sub(r" +", " ", text)
        return text
