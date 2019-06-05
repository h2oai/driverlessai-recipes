# Author: Michelle Tanco - michelle.tanco@h2o.ai
# Last Updated: May 28th, 2019
# Purpose: Functions to ease testing a new custom transformer from the python client

import pandas as pd
import os
import shutil
import sys
from h2oai_client import Client

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def connect():
    # Login info
    dai_url = "http://IPADDRESS:12345"
    dai_user = "UserName"
    dai_pwd = "Password"

    return Client(dai_url, dai_user, dai_pwd)


def print_system_custom_transformers():

    h2oai = connect()

    all_transformers = h2oai.list_transformers()

    print(all_transformers[0].dump())

    names = list(map(lambda x: x.name, all_transformers))
    types = list(map(lambda x: x.is_custom, all_transformers))

    all_trans = pd.DataFrame(
        {'name': names,
         'is_custom': types
         })

    print(list(all_trans[all_trans["is_custom"]]["name"]))


def test_debug_pyclient():

    # Data Information
    data_file_name = "****.csv"
    y = "****"

    # Transformers Information
    transformer_file_name = "****.py"

    # Use empty lists if you want to test how the transformer does in relation to other transformers
    transformers_noncustom = []
    transformers_custom_nontesting = []

    # All Official Transformers
    transformers_noncustom = ['CVCatNumEncode', 'CVTargetEncode'
        , 'CatOriginalTransformer', 'ClusterDistTransformer'
        , 'ClusterIdTransformer', 'ClusterTETransformer', 'DatesTransformer'
        , 'EwmaLagsTransformer', 'FrequentTransformer', 'InteractionsTransformer'
        , 'IsHolidayTransformer', 'LagsAggregatesTransformer', 'LagsInteractionTransformer'
        , 'LagsTransformer', 'LexiLabelEncoder', 'NumCatTETransformer', 'NumToCatTETransformer'
        , 'NumToCatWoEMonotonicTransformer', 'NumToCatWoETransformer', 'OneHotEncodingTransformer'
        , 'OriginalTransformer', 'SortedLETransformer', 'StrFeatureTransformer', 'TextClustDistTransformer'
        , 'TextClustTETransformer', 'TextLinModelTransformer', 'TextTransformer', 'TruncSVDNumTransformer'
        , 'WeightOfEvidenceTransformer']

    # Any Installed Custom Transformers you don't want to test
    transformers_custom_nontesting = ['MyLogTransformer']

    # A list of all transformers we don't want in our experiment
    all_nontest_transformers = transformers_noncustom + transformers_custom_nontesting

    # Step Zero: Connect to Driverless AI
    h2oai = connect()

    # Step One: Load Data Set

    # Get all data sets that are already loaded into DAI
    all_data_sets = h2oai.list_datasets(0, 100, include_inactive=True).datasets
    all_data_sets = pd.DataFrame({
        'key': list(map(lambda x: x.key, all_data_sets))
        , 'name': list(map(lambda x: x.name, all_data_sets))})

    print("PRE-LOADED DATASETS:")
    print(all_data_sets)

    # check if data was pre-loaded - if so use that data set - if not load data
    # this means that if your base data was changed you will need to delete the data from the UI
    # or remove it using h2oai.delete_dataset(key)
    if data_file_name in all_data_sets['name'].values:
        print("Data already loaded ", data_file_name)
        data_key = all_data_sets[all_data_sets["name"] == data_file_name]["key"][0]
        # data_load_job = h2oai.get_dataset_job(data_key).entity
    else:
        print("Loading file ", data_file_name)
        data_load_job = h2oai.upload_dataset_sync(data_file_name)
        data_key = data_load_job.key

    # Step Two: Load custom transformer
    # probably not good to just upload every time
    # no function to delete from python, only from ssh-ing in
    # rm tmp/contrib/transformers/[function]_randomletters_content.py

    print("Uploading Transformer ", transformer_file_name)
    my_transformer = h2oai.upload_custom_recipe_sync(transformer_file_name)

    # returns true or false - exit if fails - check DAI UI for error message (make new experiment & upload)
    if my_transformer:
        print("Transformer uploaded successfully")
    else:
        print("Transformer uploaded failed, exiting program.")
        sys.exit()

    # Step Three: Run experiment (and related tasks)
    print("Starting Experiment")
    experiment = h2oai.start_experiment_sync(
        dataset_key=data_key
        , target_col=y
        , is_classification=True
        , accuracy=1
        , time=1
        , interpretability=10
        , scorer="F1"
        , score_f_name=None
        , config_overrides="""
                                    feature_brain_level=0
                                    exclude_transformers={dont_use}
                                    """.format(dont_use=all_nontest_transformers)
    )

    # if you have a pre-run experiment you want to examine
    # experiment = h2oai.get_model_job("lomotare").entity

    # Step Four: Check the transformation was used

    summary_path = h2oai.download(src_path=experiment.summary_path, dest_dir=".")
    dir_path = "h2oai_experiment_summary_" + experiment.key
    import zipfile
    with zipfile.ZipFile(summary_path, 'r') as z:
        z.extractall(dir_path)

    # View Features, hopefully your transformer is here!
    features = pd.read_table(dir_path + "/features.txt", sep=',', skipinitialspace=True)
    print(features)

    # Step Five: Transform data and ensure it looks as expected
    transform = h2oai.fit_transform_batch_sync(model_key=experiment.key
                                               , training_dataset_key=data_key
                                               , validation_dataset_key=None
                                               , test_dataset_key=None
                                               , validation_split_fraction=0.25
                                               , seed=1234
                                               , fold_column=None)

    # Download the training and validation transformed data
    transform_train_path = h2oai.download(src_path=transform.training_output_csv_path, dest_dir=".")
    transform_validate_path = h2oai.download(src_path=transform.validation_output_csv_path, dest_dir=".")

    transform_train = pd.read_table(transform_train_path, sep=',', skipinitialspace=True)
    transform_validate = pd.read_table(transform_validate_path, sep=',', skipinitialspace=True)

    print(transform_train.head())
    print(transform_validate.head())

    # Step Six: Join back to your training data

    # Step Seven: Clean up
    os.remove(summary_path)
    os.remove(transform_train_path)
    os.remove(transform_validate_path)
    shutil.rmtree(dir_path)


# run this test with `pytest -s how_to_test_from_py_client.py` or `python how_to_test_from_py_client.py`
if __name__ == '__main__':
    print_system_custom_transformers()
    test_debug_pyclient()

