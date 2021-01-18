"""

This data recipe lets the user to augment new features to the dataset using the Augment Cloud Service
1. The recipe pushes the dataset to user's snowflake account
2. The recipe imports the dataset into augment service
3. The recipe submits request for augmentation
4. The recipe polls the API for the matching engine to process the request
5. The recipe submits job for the creation of new augment table. The service uses correlation to find relevance between the new features and the given target label
6. The recipe polls the API for the completion of the table creation
6. The recipe exports the dataset back to user's snowflake account
7. The recipe downloads, saves the dataset from snowflake into driverlessai instance and returns the file path
8. A new dataset is created in DAI with the augmented columns


Prerequisites:
-------------
    1. User account in snowflake
    2. User account in Augment Cloud Service
    3. Access token to execute the API calls
    
Augment API Sign up & Access Token Generation:
---------------------------------------------
    https://github.com/h2oai/data-augmentation-saas/tree/dev/cmd/api-server#how-to-sign-up-and-generate-access-tokens-to-the-sandboxplayground-environment
    
Module Setup:
------------
    API Parameters
    --------------
        API_URL(str): Base endpoint for the augment service
        API_ACCESS_TOKEN(str): Place your API access token in this variable
    Snowflake Parameters:
    --------------------
        SNOWFLAKE(dict): Update the snowflake settings in the dictionary    
    Augmentation Parameters:
    -----------------------
        TARGET_LABEL(str): The name of the target column of your dataset in the variable `TARGET_LABEL`
        MAX_CANDIDATES_TO_JOIN(int): Number of matching tables that you would like to join
        MAX_FEATURES_PER_CANDIDATE(int): Number of features that you would like to pick from each table
        MATCH_SETTINGS.match_rate(int): Percentage of rows that should match based on the join key. Optional. Default 50%
        MATCH_SETTINGS.match_columns(str): Comma separated list of columns that you prefer to look for suitable matches. Optional.

"""
import datatable as dt
import datetime
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import shutil
import random
import requests
import time
from typing import Union, List
from snowflake.connector import connect
import uuid

from h2oaicore.data import CustomData


logger = logging.getLogger(__name__)

####################################################### MODULE SETUP #################################################################
# API Token
API_ACCESS_TOKEN = "<Augment API access token>"
API_URL = "http://data-augment-playground.h2o.ai/v1"

# Snowflake Settings
SNOWFLAKE = {
    "url": "<snowflake account url>",
    "username": "<username>",
    "password": "<password>",
    "database": "<database>",
    "warehouse": "<warehouse>",
    "schema": "<schema>",
    "role": "<role name>"
}

# Augmentation Settings
TARGET_LABEL = "Sales"
MATCH_SETTINGS = {
    "match_rate":60,
    "match_columns": "Postal Code"
}
MAX_FEATURES_PER_CANDIDATE = 5
MAX_CANDIDATES_TO_JOIN = 5

########################################################## END ########################################################################


def to_column_def(df, i):
    st = df.stypes[i]
    t = ""
    if st == dt.stype.bool8:
        t = "BOOLEAN"
    elif st in (dt.stype.int8, dt.stype.int16, dt.stype.int32, dt.stype.int64):
        t = "INTEGER"
    elif st in (dt.stype.float32, dt.stype.float64):
        t = "FLOAT"
    elif st in (dt.stype.str32, dt.stype.str64):
        t = "TEXT"  # TODO use unused to handle enums
    else:  # obj64
        t = "TEXT"
    if t == "TEXT":
        try:
            values = df[df.names[i]].to_list()[0]
            cnt = 0
            for j in range(100):
                val = random.choice(values)
                if val and datetime.datetime.strptime(val, "%Y-%m-%d"):
                    cnt += 1
            if cnt > 80:
                t = "DATE"
        except ValueError:
            pass
    return f'"{df.names[i]}" {t}'


def find_stypes(df):
    cols = [to_column_def(df, i) for i in range(df.ncols)]
    return cols


def get_schema_from_csv(csv_file_path, db, schema, table_name):
    df = dt.fread(csv_file_path, na_strings=["", "NULL"])
    # add row_id to the table
    col_defs = find_stypes(df)
    ddl = """CREATE OR REPLACE TABLE "{0}"."{1}"."{2}" ({3}) """.format(
        db, schema, table_name, " , ".join(col_defs)
    )
    return ddl


class SnowflakeConnector:
    """Sourced from `https://github.com/h2oai/h2oai/blob/dev/h2oai/connectors/__init__.py`"""

    def __init__(self, url, username, password, database, warehouse, schema, role):
        self.url = url
        self.user = username
        self.password = password
        self.database = database
        self.warehouse = warehouse
        self.schema = schema
        self.client = None
        self.cursor = None
        self.role = role
        self.uid = "h2oai" + str(uuid.uuid4()).replace("-", "")
        self.account, self.region = self._parse_snowflake_url(url)

    def _initialize_client(
        self, region, database, warehouse, schema, role, sf_user, password
    ):
        try:
            kwargs = {
                "user": sf_user,
                "password": password,
                "account": self.account,
                "region": region,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "role": role,
                "application": "H2O",
            }
            self.client = connect(**kwargs)
            self.cursor = self.client.cursor()
            self.cursor.execute(f"use {database}")
            self.cursor.execute("create or replace stage {};".format(self.uid))
        except Exception as e:
            kwargs.pop("password")
            logger.exception(
                f"Error while initializing connection to snowflake with params {kwargs}"
            )
            raise Exception(
                f"Error while initializing connection to snowflake with params {kwargs}"
            )

    def _parse_snowflake_url(self, snowflake_url=""):
        url_patterns = [
            {
                "pattern": "^(http|https):\/\/([a-z0-9_-]*)[.]([a-z0-9\.-_]*)[.]snowflakecomputing[.]com$",
                "account": 2,
                "region": 3,
            },
            {
                "pattern": "^(http|https):\/\/([a-z0-9_-]*)[.]snowflakecomputing[.]com$",
                "account": 2,
                "region": -1,
            },
            {
                "pattern": "^(http|https):\/\/([a-z0-9]*)[.]([a-z0-9\.-_]*)[.]([a-z0-9\.-_]*)[.]snowflakecomputing[.]com$",
                "account": 2,
                "region": 3,
            },
        ]
        account = ""
        region = ""
        for pattern in url_patterns:
            reg_match = re.match(pattern["pattern"], snowflake_url)
            if reg_match:
                account = reg_match.group(pattern["account"])
                if (pattern["region"]) != -1:
                    region = reg_match.group(pattern["region"])
                break
        return account, region

    def import_data(
        self,
        database,
        warehouse,
        query,
        dst,
        schema,
        role="",
        region="",
        optional_file_formatting="",
        sf_user="",
        sf_pass="",
    ):
        if not sf_user:
            sf_user = self.user
        if not sf_pass:
            sf_pass = self.password
        if not role:
            role = self.role
        if not region:
            region = self.region

        self._initialize_client(
            region, database, warehouse, schema, role, sf_user, sf_pass
        )

        try:
            filename = dst.split("/")[-1]
            filepath = dst.replace(filename, "")
            exp_name = filename.replace(".csv", "")
            outpath = os.path.join(filepath, exp_name)

            if not os.path.exists(outpath):
                os.makedirs(outpath)
            copy_query = """
                                COPY INTO @{}/{}
                                FROM ({})
                                FILE_FORMAT = (TYPE="CSV" COMPRESSION="NONE" FIELD_OPTIONALLY_ENCLOSED_BY='"' {})
                                OVERWRITE = TRUE
                                HEADER = TRUE
                                """.format(
                self.uid, exp_name, query, optional_file_formatting
            )
            copy_query = copy_query.replace("\n", " ")
            self.cursor.execute(copy_query)

            # Create a list of all the files generated by the query in Snowflake stage
            files_list = []
            for row in self.cursor.execute("list @{}".format(self.uid)):
                if exp_name in row[0]:
                    csv_file = row[0]
                    files_list.append(csv_file)

            # Snowflake didn't return any data - exiting
            if not files_list:
                raise ValueError(
                    """Error while attempting to query snowflake: query returned no data.
This could be a result of a table with no rows or certain filter clauses such as `WHERE` or `LIMIT`.
Please verify the query and run again or check Snowflake directly for information on executed processes."""
                )

            # Download all files from Snowflake stage associated to the user query
            for csv_file in files_list:
                self.cursor.execute("get @{} file://{};".format(csv_file, outpath))
            # cleanup: close cursor drop temporary stage
            self.cursor.execute("drop stage {};".format(self.uid))
            self.cursor.close()

            return outpath

        except Exception as e:
            logger.exception(f"Unable to import query {query} into file due to {e}")
            self.cursor.execute("drop stage {};".format(self.uid))
            self.cursor.close()
            if os.path.isdir(outpath):
                shutil.rmtree(outpath)
            raise Exception(f"Unable to import query {query} into file due to {e}")

    def export_data(
        self,
        file_path,
        database,
        warehouse,
        schema,
        table_name,
        role="",
        region="",
        sf_user="",
        sf_pass="",
    ):
        if not sf_user:
            sf_user = self.user
        if not sf_pass:
            sf_pass = self.password
        if not role:
            role = self.role
        if not region:
            region = self.region

        self._initialize_client(
            region, database, warehouse, schema, role, sf_user, sf_pass
        )
        try:
            files = os.listdir(file_path)
            # Create the table
            create_table_query = get_schema_from_csv(
                os.path.join(file_path, files[0]), self.database, self.schema, table_name
            )
            self.cursor.execute(create_table_query)
            # Create csv file format
            self.cursor.execute(
                f"""
                CREATE OR REPLACE FILE FORMAT csvformat_{role.lower()}
                TYPE = "CSV"
                FIELD_DELIMITER = ","
                SKIP_HEADER = 1
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            """
            )
            self.cursor.execute(
                """
                PUT file://{}/*.csv @{} auto_compress=true;
            """.format(
                    file_path, self.uid
                )
            )
            # Load the data from stage into the table
            self.cursor.execute(
                """
                COPY INTO "{}"."{}"."{}"
                FROM @{}/{}.gz
                FILE_FORMAT = (format_name = csvformat_{})
                ON_ERROR = 'skip_file'
            """.format(
                    self.database, self.schema, table_name, self.uid, files[0], role.lower()
                )
            )
            logger.info(self.cursor.fetchall())
        except Exception as e:
            logger.exception(f"Unable to export file {file_path} into table due to {e}")
            self.cursor.execute("drop stage {};".format(self.uid))
            self.cursor.close()
            raise Exception(f"Unable to export file {file_path} into table due to {e}")
        

class APIConnector:
    def __init__(self, endpoint, access_token):
        self.endpoint = endpoint
        self.access_token = access_token
        

class DatasetAPI:
    """Class to import, export and list user datasets"""
    def __init__(self, connection):
        self.__connection = connection
        
    def import_data(self, display_name, import_url, username, password, database, warehouse, schema, role, table_name):
        """Method to import data from user snowflake to h2o snowflake account
        
        Args:
            display_name (str): Name of the dataset for API use
            import_url (str): Snowflake account url
            username (str): Snowflake username
            password (str): Snowflake password
            database (str): Name of the snowflake database
            warehouse (str): Name of the snowflake warehouse
            schema (str): Name of the snowflake schema
            role (str): Role for the specific user
            table_name (str): Name of the table to be imported
            
        Returns:
            tuple(job_id, error)
        """
        try:
            url = f"{self.__connection.endpoint}/sql/import"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            params = {
                "import_sql": {
                    "display_name": display_name,
                    "connection_url": import_url,
                    "username": username,
                    "password": password,
                    "database": database,
                    "warehouse": warehouse,
                    "schema": schema,
                    "role": role,
                    "table_name": table_name
                }
            }
            resp = requests.post(url, data=json.dumps(params), headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            return result["id"], None
        except Exception as e:
            return None, str(e)
            
    
    def export_data(self, dataset_id, export_url, username, password, database, warehouse, schema, role):
        """Method to export data from h2o snowflake to user snowflake account
        
        Args:
            dataset_id (str): Name of the dataset to be exported
            export_url (str): Snowflake account url
            username (str): Snowflake username
            password (str): Snowflake password
            database (str): Name of the snowflake database
            warehouse (str): Name of the snowflake warehouse
            schema (str): Name of the snowflake schema
            role (str): Role for the specific user
            
        Returns:
            tuple(job_id, error)
        """
        try:
            url = f"{self.__connection.endpoint}/sql/export"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            params = {
                "export_sql": {
                    "connection_url": export_url,
                    "username": username,
                    "password": password,
                    "database": database,
                    "warehouse": warehouse,
                    "schema": schema,
                    "role": role,
                    "dataset_id": dataset_id
                }
            }
            resp = requests.post(url, data=json.dumps(params), headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            return result["id"], None
        except Exception as e:
            return None, str(e)
            
    def check_job_status(self, job_id):
        """Method to check the status of the import/export job
        
        Args:
            job_id (str): Uuid of the job
            
        Returns:
            tuple(status, error)
        """
        try:
            url = f"{self.__connection.endpoint}/sql/{job_id}"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            resp = requests.get(url, headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            return result["status"], None
        except Exception as e:
            return "", str(e)
    
    def retrieve(self):
        """Method to retrieve the list of datasets present in user account
            
        Returns:
            list of datasets
        """
        try:
            url = f"{self.__connection.endpoint}/dataset"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            resp = requests.get(url, headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            return result, None
        except Exception as e:
            return [], str(e)

    
class AugmentAPI:
    """Class to submit augmentation requests"""
    def __init__(self, connection, dataset_id, display_name, match_settings=None):
        self.__connection = connection
        self.dataset_id = dataset_id
        self.name = display_name
        self.validate(match_settings or {})
        self.id = None
        self.join_job_id = None
        self.__matches = None
        
    def validate(self, params):
        """Method to validate parameters"""
        if not isinstance(params.get("match_rate", 0), int):
            raise ValueError("Incorrect format for `match_rate`")
        if not isinstance(params.get("fuzzy_threshold", 0), int):
            raise ValueError("Incorrect format for `fuzzy_threshold`")
        self.params = {
            "augment": {
                "display_name": self.name,
                "dataset_id": self.dataset_id,
                "settings": params or {}
            }
        }
    
    def initialise(self):
        """Method that submits the augmentation request"""
        try:
            url = f"{self.__connection.endpoint}/augment"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            logger.info(self.params)
            resp = requests.post(url, data=json.dumps(self.params), headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            self.id = result["id"]
            return self.id, None
        except Exception as e:
            return None, str(e)
    
    @property
    def status(self):
        """Returns the status of the augmentation requrest"""
        if not self.id:
            return "", "Request not initialized"
        try:
            url = f"{self.__connection.endpoint}/augment/{self.id}"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            resp = requests.get(url, headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            return result["status"], None
        except Exception as e:
            return "", str(e)
        
        
    def validate_request_status(self):
        """Validates the status of the augmentation request. For internal use"""
        if not self.id:
            return False, "Request not initialized"
        
        if self.status[0] != "MATCHING_COMPLETED":
            return False, f"Invalid request status {self.status[0]}"
        
        return True, None
    
    @property
    def candidates(self):
        """Returns the list of matching candidate datasets and their details"""
        flag, err = self.validate_request_status()
        if not flag:
            return [], err
        
        if not self.__matches:
            try:
                url = f"{self.__connection.endpoint}/augment/{self.id}/candidates"
                headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
                resp = requests.get(url, headers=headers, verify=False)
                resp.raise_for_status()
                result = resp.json()
                self.__matches = result
                return result, None
            except Exception as e:
                return [], str(e)
        return self.__matches, None
    
    def apply_join(self, target_label, max_candidates_to_join, max_features_per_candidate):
        """Method to submit the final request to create the augmented table
        
        Args:
            target_label (str): Column to be predicted from the input table
            max_candidates_to_join (int): No of candidates to pick for the join
            max_features_per_candidate (int): No of features to pick from each candidate
            
        Returns:
            tuple(job_id, error)
        """
        flag, err = self.validate_request_status()
        if not flag:
            return None, err
        
        try:
            url = f"{self.__connection.endpoint}/augment/{self.id}/match"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            params = {
                "match_job": {
                    "max_features_per_result": max_features_per_candidate,
                    "max_results": max_candidates_to_join,
                    "target_label": target_label
                }
            }
            resp = requests.post(url, data=json.dumps(params), headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            self.join_job_id = result["id"]
            return result["id"], None
        except Exception as e:
            return None, str(e)
    
    @property
    def join_status(self):
        """Returns the status of the join request"""
        flag, err = self.validate_request_status()
        if not flag:
            return "", err
        
        if not self.join_job_id:
            return "", "Join not applied"
        
        try:
            url = f"{self.__connection.endpoint}/augment/{self.id}/match/{self.join_job_id}"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            resp = requests.get(url, headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            return result["status"], None
        except Exception as e:
            return "", str(e)

    @property
    def output(self):
        """Returns the dataset id of the newly created augmented table"""
        flag, err = self.validate_request_status()
        if not flag:
            return "", err
        
        if not self.join_job_id:
            return "", "Join not applied"
        
        if self.join_status[0] != "DONE":
            return "", "Join operation not completed yet"
        
        try:
            url = f"{self.__connection.endpoint}/augment/{self.id}/match/{self.join_job_id}"
            headers = {"Authorization": f"Bearer {self.__connection.access_token}"}
            resp = requests.get(url, headers=headers, verify=False)
            resp.raise_for_status()
            result = resp.json()
            return result["dataset_id"], None
        except Exception as e:
            return "", str(e)


class AugmentDataset(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        # exit gracefully if method is called as a data upload rather than data modify
        if X is None:
            return []
        
        from h2oaicore.systemutils import config
        
        orig_uuid = str(uuid.uuid4())
        temp_path = os.path.join(config.data_directory, config.contrib_relative_directory, "data", orig_uuid)
        # Save files to disk
        os.makedirs(temp_path, exist_ok=True)
        orig_file = os.path.join(temp_path, orig_uuid + ".csv")
        X = dt.Frame(X).to_pandas()
        X.to_csv(orig_file, index=False)
        
        # Initialize Snowflake connection
        sf = SnowflakeConnector(**SNOWFLAKE)
        # Push file to snowflake database
        sf.export_data(temp_path, SNOWFLAKE["database"], SNOWFLAKE["warehouse"], SNOWFLAKE["schema"], orig_uuid)
        # Initialise Augment API Conector
        conn = APIConnector(API_URL, API_ACCESS_TOKEN)
        # Import your dataset into the API system
        data_handle = DatasetAPI(conn)
        job_id, err = data_handle.import_data(
            orig_uuid,
            SNOWFLAKE["url"],
            SNOWFLAKE["username"],
            SNOWFLAKE["password"],
            SNOWFLAKE["database"],
            SNOWFLAKE["warehouse"],
            SNOWFLAKE["schema"],
            SNOWFLAKE["role"],
            orig_uuid
        )
        if err:
            raise Exception(f"Unable to import data into Augment Service due to {err}")
        logger.info(f"Augment Recipe: Data Import: {job_id} {err}")
        # Wait for a while to give the import job time to complete
        # Or check for status using the below call
        for i in range(3):
            time.sleep(30)
            status, err = data_handle.check_job_status(job_id)
            if status == "DONE":
                break
        else:
            raise Exception(f"Unable to import data into Augment Service due to {err}")
        # Once complete, retrieve the list to verify the same
        datasets, err = data_handle.retrieve()
        if err:
            raise Exception(f"Unable to retrieve dataset list for the user due to {err}")
        input_dataset_id = [i["id"] for i in datasets if i["table_name"] == orig_uuid][0]
        # Submit augment request for the input dataset
        my_augment_request = AugmentAPI(conn, input_dataset_id, f"augment request for {orig_uuid}", MATCH_SETTINGS)
        req_id, err = my_augment_request.initialise()
        logger.info(f"Augment Recipe: Submit Request: {req_id} {err}")
        if err:
            raise Exception(f"Unable to submit augmentation request to the service due to {err}")
        # Wait for the matching to complete or alternative check status using 
        for i in range(5):
            time.sleep(30)
            status, err = my_augment_request.status
            if status == "MATCHING_COMPLETED":
                break
        else:
            raise Exception(f"Error while finding matches for the given dataset due to {err}")
        # Retrieve list of matching candidate datasets
        candidates = my_augment_request.candidates
        # Submit final job to join the tables
        my_augment_request.apply_join(target_label=TARGET_LABEL, max_candidates_to_join=MAX_CANDIDATES_TO_JOIN, max_features_per_candidate=MAX_FEATURES_PER_CANDIDATE)
        # Wait for the join to complete
        for i in range(20):
            time.sleep(30)
            status, err = my_augment_request.join_status
            if status == "DONE":
                break
        else:
            raise Exception(f"Error while creating the augmented table due to {err}")
        # Retrieve augmented table id
        augment_dataset_id, err = my_augment_request.output
        logger.info(f"Augment Recipe: Augmented DatasetId: {augment_dataset_id}")
        if err:
            raise Exception(f"Error while creating the augmented table due to {err}")

        # Export augmented data to your snowflake
        job_id, err = data_handle.export_data(
            augment_dataset_id,
            SNOWFLAKE["url"],
            SNOWFLAKE["username"],
            SNOWFLAKE["password"],
            SNOWFLAKE["database"],
            SNOWFLAKE["warehouse"],
            SNOWFLAKE["schema"],
            SNOWFLAKE["role"],
        )
        if err:
            raise Exception(f"Unable to export data to user's snowflake account due to {err}")
        # Check job status
        for i in range(3):
            time.sleep(30)
            status, err = data_handle.check_job_status(job_id)
            if status == "DONE":
                logger.info(f"Augment Recipe: Augmented dataset exported successfully")
                break
        else:
            raise Exception(f"Error while exporting data to user's snowflake account due to {err}")
        
        datasets, err = data_handle.retrieve()
        augment_table_name = [i["table_name"] for i in datasets if i["id"] == augment_dataset_id][0]
        logger.info(f"Augment Recipe: Augment Table Name: {augment_table_name}")
        # Push file to snowflake database
        file_train = f"{temp_path}/{augment_table_name}.csv"
        sf.import_data(
            SNOWFLAKE["database"],
            SNOWFLAKE["warehouse"],
            f'SELECT * FROM "{SNOWFLAKE["database"]}"."{SNOWFLAKE["schema"]}"."{augment_table_name}"',
            file_train,
            SNOWFLAKE["schema"],
            SNOWFLAKE["role"]
        )
        outpath = os.path.realpath(f"{temp_path}/{augment_table_name}")
        logger.info(f"Augment Recipe: Imported file from snowflake {outpath}")
        return [f"{outpath}/{i}" for i in os.listdir(outpath)]
            