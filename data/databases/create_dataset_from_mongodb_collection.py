"""Create dataset from MonogDB"""

# Author: Nicholas Png
# Created: 31/01/2020
# Last Updated: 31/01/2020

import datatable as dt
import pandas as pd
from h2oaicore.data import CustomData


_global_modules_needed_by_name = ["pymongo"]
# Please fill before usage
# Note that this information is logged in Driverless AI logs.
MONGO_HOST_IP = "127.0.0.1"
MONGO_PORT = "27017"
MONGO_USERNAME = "h2oai"
MONGO_PASSWORD = "h2oai"
MONGO_DB = "test"
MONGO_COLLECTION = "creditcardusers"


class MongoDbData(CustomData):

    _modules_needed_by_name = ["pymongo"]

    @staticmethod
    def create_data(X: dt.Frame = None):
        from pymongo import MongoClient

        # Initialize MongoDB python client
        connection_string = f"mongodb://{MONGO_PASSWORD}:{MONGO_USERNAME}@{MONGO_HOST_IP}:{MONGO_PORT}"
        client = MongoClient(connection_string)

        # Use MongoDB python client to obtain list of all documents in a specific database + collection
        db = client.get_database(MONGO_DB)
        coll = db.get_collection(MONGO_COLLECTION)
        docs = coll.find()

        # Convert MongoDB documents cursor to pandas dataframe
        df = pd.DataFrame.from_dict(docs)

        # Cast "_id" column as string since datatable cannot accept arbitrary objects
        df["_id"] = df["_id"].astype(str)

        # return dict where key is name of dataset and value is a datatable Frame of the data.
        return {"mongodb_import": dt.Frame(df)}
