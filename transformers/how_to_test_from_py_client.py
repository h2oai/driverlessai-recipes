# Edited by: Michelle Tanco
# Last update: 05/19/2019
# Example script for testing transformers in DAI

import pandas as pd
import os
import shutil
import sys


def test_debug_pyclient():
    from h2oai_client import Client

    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    # Login info
    dai_url = "http://****:12345"
    dai_user = "h2oai"
    dai_pwd = "****"

    # Data Information
    data_file_name = "****.csv"
    y = "****"

    # Transformers information
    transformer_file_name = "****.py"

    transformers_noncustom = []
    transformers_custom_nontesting = []

    # All Offical Transformers
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

    all_nontest_transformers = transformers_noncustom + transformers_custom_nontesting

    # STEP ZERO: Connect to Driverless AI
    h2oai = Client(dai_url, dai_user, dai_pwd)

    # STEP ONE: Load data set (and related tasks)

    # view all data sets in DAI
    all_data_sets = h2oai.list_datasets(0, 100)
    all_data_sets = pd.DataFrame({
        'key': list(map(lambda x: x.key, all_data_sets))
        , 'name': list(map(lambda x: x.name, all_data_sets))})

    print("PRE-LOADED DATASETS:")
    print(all_data_sets)

    # check if data was pre-loaded - if so use that data set - if not load data
    if data_file_name in all_data_sets['name'].values:
        print("\nData already loaded ", data_file_name)
        data_key = all_data_sets[all_data_sets["name"] == data_file_name]["key"][0]
        data_load_job = h2oai.get_dataset_job(data_key).entity
    else:
        print("\nLoading file ", data_file_name)
        data_load_job = h2oai.upload_dataset_sync(data_file_name)
        data_key = data_load_job.key

    # STEP TWO: Load custom transformer (and related tasks)
    # probably not good to just upload every time
    # no function to delete from python, only from ssh-ing in
    # rm tmp/contrib/transformers/[function]_randomletters_content.py

    print("\nUploading Transformer ", transformer_file_name)
    my_transformer = h2oai.upload_custom_recipe_sync(transformer_file_name)

    # returns true or false - exit if fails - check DAI UI for error message
    if my_transformer:
        print("\nTransformer uploaded successfully\n")
    else:
        print("\nTransformer uploaded failed, exiting program.\n")
        sys.exit()

    # STEP THREE: Run experiment (and related tasks)
    print("\nStarting Experiment\n")
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

    # experiment = h2oai.get_model_job("lomotare").entity

    # STEP FOUR: Check the transformation was used

    # Download Summary
    summary_path = h2oai.download(src_path=experiment.summary_path, dest_dir=".")
    dir_path = "h2oai_experiment_summary_" + experiment.key
    import zipfile
    with zipfile.ZipFile(summary_path, 'r') as z:
        z.extractall(dir_path)

    # View Features
    features = pd.read_table(dir_path + "/features.txt", sep=',', skipinitialspace=True)
    print(features)

    # STEP FIVE: Transform data and ensure it looks as expected
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

    # STEP 1000: Clean up
    os.remove(summary_path)
    os.remove(transform_train_path)
    os.remove(transform_validate_path)
    shutil.rmtree(dir_path)


# run this test with `pytest -s how_to_test_from_py_client.py` or `python how_to_test_from_py_client.py`
if __name__ == '__main__':
    test_debug_pyclient()
