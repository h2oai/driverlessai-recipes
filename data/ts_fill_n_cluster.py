"""Data Recipe to fill missing values in TS data and then create new data sets from TS Clustering"""

"""
Date should be pre-loaded in Driverless AI

__version__ = 0.1

authored by @mtanco (Michelle Tanco)


Required User Defined Inputs: target column, date column, group by column(s)
Optional User Defined Inputs: number of clusters

"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

""" 
	""User Inputs "" 

	| variable             | type | default | description                                             |
	| date_column          | str  | NA      | Column to order the data by, would be used as TS column 
	|									  in Driverless AI Modeling           					  | 
	| group_by_columns     | list | NA      | Columns that define each series, would be Target Group
	|									  Columns in Driverelss AI Modleing						  |
	| y		    	   	   | str  | NA   	| TS Column to cluster - what you would forecast in DAI   |
	| num_clusters     	   | int  | 3       | Number of clusters to generate in the output            | 

"""

# defaults are for the Walmart_Train Dataset
date_column = "Date"
group_by_columns = ["Store", "Dept"]
y = "Weekly_Sales"
num_clusters = 3


class KMeansClustering(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        if X is None:
            return []

        # check the datatype of user-defined columns
        if not isinstance(date_column, str):
            raise ValueError("Variable: 'date_column' should be <str>")
        if not isinstance(group_by_columns, list):
            raise ValueError("Column: 'group_by_columns' should be <list>")
        if not isinstance(y, str):
            raise ValueError("Column: 'y' should be <str>")
        if not isinstance(num_clusters, int):
            raise ValueError("Column: 'num_clusters' should be <int>")

        # check if user-defined inputs exist in the dataset
        features = list(X.names)

        if date_column not in features:
            raise ValueError("Column: '" + date_column + "' is not present in the data set")
        if y not in features:
            raise ValueError("Column: '" + y + "' is not present in the data set")
        for _ in group_by_columns:
            if _ not in features:
                raise ValueError("Group by Column: '" + str(_) + "' is not present in the dataset")

        # Order by groups + time
        df_train = X.to_pandas()
        df_train = df_train.sort_values(group_by_columns + [date_column])

        # Fill any Missing Values with 0 - this will fill Y and any X predictors (non group or date) with 0
        unique_dates = pd.DataFrame(df_train[date_column].unique(), columns=[date_column])
        unique_dates['key'] = 0

        unique_groups = df_train[group_by_columns].drop_duplicates()
        unique_groups['key'] = 0

        all_vals = pd.merge(unique_dates, unique_groups, how="outer").drop("key", axis=1)
        df_train_filled = pd.merge(df_train, all_vals, how="outer")
        df_train_filled[y] = df_train_filled[y].fillna(0)

        # Prepare one row per series of all target column - clustering only on the time series
        ts_arrays = df_train_filled.groupby(group_by_columns).agg(list)[y].to_list()

        # Create clusters
        km = KMeans(n_clusters=num_clusters, random_state=42)

        c_ids = km.fit_predict(ts_arrays)
        cluster_ids = pd.DataFrame(c_ids, columns=["Cluster_ID"])
        cluster_ids = pd.concat([cluster_ids.reset_index(drop=True),
                                 unique_groups[group_by_columns].reset_index(drop=True)], axis=1)

        cluster_df = pd.merge(df_train_filled, cluster_ids)

        datas_to_model = {"TS_Filled_Values_All": cluster_df}
        for c in cluster_df["Cluster_ID"].unique():
            nm = "TS_Filled_Values_Clust" + str(c)
            datas_to_model.update({nm: cluster_df[cluster_df["Cluster_ID"] == c]})

        return datas_to_model
