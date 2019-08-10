"""Testing a BYOR Transformer the PyClient - works on 1.7.0 & 1.7.1-17"""
import pandas as pd
from h2oai_client import Client
import sys
import zipfile
import os
import shutil

# TODO: re-write the already uploaded data check to account for numpy warning of type mismatch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Print and Debug Nicely
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# The following are parameters that need to be set to run these functions
# TODO: to redo this is a nicer way

# Connect to Driverless AI
h2oai = Client('', '', '')

# Data Information
data_file_name = ""
data_file_location = "" + data_file_name
y = ""

# Transformers Information
transformer_name = ""
transformer_file_name = ""
transformer_file_location = "" + transformer_file_name

# Location to Download Files
download_file_location = ""


# Print the default & custom transformers on the system, return list of all transformers
def get_transformers(print_details=True):
    all_transformers = h2oai.list_transformers()

    names = list(map(lambda x: x.name, all_transformers))
    types = list(map(lambda x: x.is_custom, all_transformers))

    all_trans = pd.DataFrame({
        'name': names,
        'is_custom': types
    })

    if print_details:
        print("GET TRANSFORMERS: ")
        print("\tCustom Transformers:", list(all_trans[all_trans["is_custom"]]["name"]))
        print("\tDefault Transformers:", list(all_trans[~all_trans["is_custom"]]["name"]))
        print("")

    return list(all_trans["name"])


# Load the custom transformer, exit gracefully if it fails
# TODO: return error message or logs or if it fails
def load_transformer(print_details=True):
    my_transformer = h2oai.upload_custom_recipe_sync(transformer_file_location)

    # returns true or false - exit if fails - check DAI UI for error message (make new experiment & upload)
    if my_transformer:
        if print_details:
            print("LOAD TRANSFORMER:")
            print("\tTransformer uploaded successfully")
            print("")
    else:
        print("LOAD TRANSFORMER:")
        print("\tTransformer uploaded failed, exiting program.")
        sys.exit()


# Load data if it's not already on the system, return the data set key
# TODO: re-write the already uploaded check to account for numpy warning of type mismatch
def load_data(print_details=True):
    all_data_sets = h2oai.list_datasets(0, 100, include_inactive=True).datasets
    all_data_sets = pd.DataFrame({
        'key': list(map(lambda x: x.key, all_data_sets))
        , 'name': list(map(lambda x: x.name, all_data_sets))})

    if data_file_name in all_data_sets['name'].values:
        # [0] is used so we get a sting and not a pandas.core.series.Series
        dai_dataset_key = all_data_sets[all_data_sets["name"] == data_file_name]["key"][0]
    else:
        data_load_job = h2oai.upload_dataset_sync(data_file_location)
        dai_dataset_key = data_load_job.key

    if print_details:
        print("LOAD DATA: ")
        print("\tExisting data on the system:")
        print(all_data_sets)
        print()
        print("\tData key for Experiment: ", dai_dataset_key)
        print()

    return dai_dataset_key


# Run an experiment on the fastest settings with only the transformer we are using
# TODO: test what happens if transformer is included with overrides but has hardcoded settings above 1/1/10
# TODO: currently assumes classification problem
# TODO: download logs if it fails
# TODO: speed up by turning off shift detection, python scoring pipeline etc. etc.
def run_test_experiment(dai_dataset_key, print_details=True):
    if print_details:
        print("RUN TEST EXPERIMENT:")
        print("\tStarting Experiment")

    experiment = h2oai.start_experiment_sync(
        dataset_key=dai_dataset_key
        , target_col=y
        , is_classification=True
        , accuracy=1
        , time=1
        , interpretability=10
        , scorer="F1"
        , score_f_name=None
        , config_overrides="included_transformers=['" + transformer_name + "']"
    )

    if print_details:
        print("\tExperiment key: ", experiment.key)
        print()

    return experiment.key


# Print all features of the final model by downloading the experiment summary
# TODO: should error or warning if our BYOR Transformer isn't there - in theory it should always be the only feature
def print_model_features(dai_experiment_key, delete_downloads=True):
    experiment = h2oai.get_model_job(dai_experiment_key).entity

    summary_path = h2oai.download(src_path=experiment.summary_path, dest_dir=download_file_location)
    dir_path = "h2oai_experiment_summary_" + experiment.key

    with zipfile.ZipFile(summary_path, 'r') as z:
        z.extractall(dir_path)

    features = pd.read_csv(dir_path + "/features.txt", sep=',', skipinitialspace=True)
    print("PRINT MODEL FEATURES:")
    print(features)
    print()

    # Delete downloaded files
    if delete_downloads:
        os.remove(summary_path)
        shutil.rmtree(dir_path)


# Print the results of the BYOR transformer on your dataset
# TODO: have only tested using the same dataset in train and validaiont on non-validated needed transformers
def print_transformed_data(dai_experiment_key, dai_dataset_key, delete_downloads=True):
    # We train and validate on the same data to get back all of th rows in the right order in transform_train
    transform = h2oai.fit_transform_batch_sync(model_key=dai_experiment_key
                                               , training_dataset_key=dai_dataset_key
                                               , validation_dataset_key=dai_dataset_key
                                               , test_dataset_key=None
                                               , validation_split_fraction=0
                                               , seed=1234
                                               , fold_column=None)

    transform_train_path = h2oai.download(src_path=transform.training_output_csv_path, dest_dir=download_file_location)

    transform_train = pd.read_csv(transform_train_path, sep=',', skipinitialspace=True)

    print("PRINT TRANSFORMED DATA:")
    print(transform_train.head(10))
    print()

    # Delete downloaded files
    if delete_downloads:
        os.remove(transform_train_path)


# run this test with `pytest -s how_to_test_from_py_client.py` or `python how_to_test_from_py_client.py`
if __name__ == '__main__':
    dai_transformer_list = get_transformers()
    load_transformer()

    data_key = load_data()
    experiment_key = run_test_experiment(data_key)

    print_model_features(experiment_key)
    print_transformed_data(experiment_key, data_key)
