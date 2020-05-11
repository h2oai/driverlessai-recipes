"""Data recipe to perform topic modeling"""
from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

# number of topics in the topic model
n_topics = 4
# text column name on which the topic modeling will be run
text_colname = "text"
# output dataset name
output_dataset_name = "df_topic_modeling"
# number of top words to be represented in the column name
n_words_colname = 10

_global_modules_needed_by_name = ["gensim==3.8.0"]

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'from', 'subject', 're', 'edu', 'use']

class LdaTopicsClass(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        # exit gracefully if method is called as a data upload rather than data modify
        if X is None:
            return []
        import os
        from h2oaicore.systemutils import config
        import gensim
        from gensim import corpora
        from gensim.utils import simple_preprocess

        X = dt.Frame(X)
        documents = X.to_pandas()[text_colname].astype(str).fillna("NA").values
        new_X = [[w for w in simple_preprocess(doc, deacc=True) if w not in stop_words] for doc in documents]

        bigram = gensim.models.Phrases(new_X, min_count=5, threshold=10)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        new_X = [bigram_mod[d] for d in new_X]

        dictionary = corpora.Dictionary(new_X)
        new_X = [dictionary.doc2bow(doc) for doc in new_X]

        model = gensim.models.ldamodel.LdaModel(new_X,
                                                num_topics=n_topics,
                                                id2word=dictionary,
                                                passes=10,
                                                alpha="auto",
                                                per_word_topics=True,
                                                random_state=2019)
        df = pd.concat([X.to_pandas(), pd.DataFrame(model.inference(new_X)[0])], axis=1)

        topics = model.print_topics(num_words=n_words_colname)
        topic_names = ["_".join([v.split("*")[1].strip('"') for v in x[1].split(" + ")]) for x in topics]
        df.columns = list(X.names) + topic_names

        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)

        # Save files to disk
        file_train = os.path.join(temp_path, output_dataset_name + ".csv")
        df.to_csv(file_train, index=False)

        return [file_train]


