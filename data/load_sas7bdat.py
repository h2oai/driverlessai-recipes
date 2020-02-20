"""
Data Recipe to load a single sas file

__version__ = 0.1

authored by @mtanco (Michelle Tanco)

Required User Defined Inputs: name of file to load

"""

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd

_global_modules_needed_by_name = ['sas7bdat']
from sas7bdat import SAS7BDAT

"""
	""User Inputs ""

	| variable		| type	| default	| description 								|
	| dai_file_path	| str	| NA		| Location on the DAI server of SAS files	|
	| file_names	| list  |       	| List of SAS files to upload 				|


"""
# defaults are datasets from:
# 1. mkdir SAS in the DAI data folder
# 2. wget http://www.principlesofeconometrics.com/zip_files/sas.zip
# 3. unzip sas.zip

dai_file_path 		 = "/data/SAS/"
file_names		 	 = ["cloth.sas7bdat", "clothes.sas7bdat"]

# TODO: default to all files in the folder
class KMeansClustering(CustomData):
	@staticmethod
	def create_data(X: dt.Frame = None) -> Union[str, List[str],
												 dt.Frame, List[dt.Frame],
												 np.ndarray, List[np.ndarray],
												 pd.DataFrame, List[pd.DataFrame]]:

		# check the datatype of user-defined columns
		if not isinstance(dai_file_path, str):
			raise ValueError("Variable: 'dai_file_path' should be <str>")
		if not isinstance(file_names, list):
			raise ValueError("Column: 'file_names' should be <list>")

		# TODO: add checs that files exist
		data_sets = {}

		for f in file_names:
		    full_path_to_file = dai_file_path + f
		    with SAS7BDAT(full_path_to_file, skip_header=False) as reader:
		        X = reader.to_data_frame()
		        print(X.head())
		        data_sets.update({f: X})

		return data_sets

