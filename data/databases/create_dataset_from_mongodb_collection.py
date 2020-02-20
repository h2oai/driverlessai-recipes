"""Create dataset from MonogDB"""

# Author: Nicholas Png
# Created: 31/01/2020
# Last Updated: 20/02/2020

import datatable as dt
import pandas as pd
from h2oaicore.data import CustomData


_global_modules_needed_by_name = ["pymongo", "dnspython"]

# Please fill before usage
# Note that this information is logged in Driverless AI logs.
MONGO_CONNECTION_STRING = "mongodb+srv://<username>:<password>@host[/[database][?options]]"
MONGO_DB = "sample_mflix"
MONGO_COLLECTION = "theaters"
DATASET_NAME = "sample_mflix.theaters"


class MongoDbData(CustomData):

    _modules_needed_by_name = ["pymongo", "dnspython"]

    @staticmethod
    def create_data(X: dt.Frame = None):
        from pymongo import MongoClient

        # Initialize MongoDB python client        
        client = MongoClient(MONGO_CONNECTION_STRING)

        # Use MongoDB python client to obtain list of all documents in a specific database + collection
        db = client.get_database(MONGO_DB)
        coll = db.get_collection(MONGO_COLLECTION)
        docs = coll.find()

        # Convert MongoDB documents cursor to pandas dataframe
        df = pd.DataFrame.from_dict(docs)

        # Cast all object columns as string since datatable cannot accept arbitrary objects
        object_cols = df.select_dtypes(include=['object']).columns
        df[object_cols] = df[object_cols].astype(str)

        # return dict where key is name of dataset and value is a datatable Frame of the data.
        return {DATASET_NAME: dt.Frame(df)}
