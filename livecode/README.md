# Live Code Templates for H2O Driverless AI Datasets

## About Driverless AI and BYOR 
See main [README](https://github.com/h2oai/driverlessai-recipes/blob/master/README.md)

## About Data Recipes
Driverless AI allows you to create a new dataset by modifying an existing dataset with a data recipe. 
When inside **Dataset Details** page:

* click the **Modify by Recipe** button in the top right portion of the UI
* click **Live Code** from the submenu that appears
* enter the code inside the **Dialog Box** that appears featurng an editor 
* enter the code for the data recipe you want to use to modify the dataset. 

The list below contains the live code templates for various applications of data recipes. Each template is designed and documented to be applied 
after modifications specific to a dataset it applies to.

## Reference Guide
* [Adding a Data Recipe](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/custom-recipes-data-recipes.html#adding-a-data-recipe)
* [Templates](https://github.com/h2oai/driverlessai-recipes/blob/master/FAQ.md#references)
* [Technical Architecture Diagram](https://raw.githubusercontent.com/h2oai/driverlessai-recipes/master/reference/DriverlessAI_BYOR.png)

## Sample Recipes
### Count: 38
  * [add\_columns\_with\_rand\_values.py](./add_columns_with_rand_values.py) ["""Augment dataset with columns containing random values"""]  

  * [aggregate\_and\_augment\_one\_to\_many.py](./aggregate_and_augment_one_to_many.py) ["""Augment data based on one-to-many relationship by means of aggregate and join"""]  

  * [aggregate\_columns\_with\_groupby.py](./aggregate_columns_with_groupby.py) ["""Group by and aggregate dataset"""]  

  * [balance\_data.py](./balance_data.py) ["""Downsample majority class in imbalanced dataset"""]  

  * [bind\_2\_datasets.py](./bind_2_datasets.py) ["""Bind 2 datasets by rows (union)"""]  

  * [bind\_X\_and\_Y.py](./bind_X_and_Y.py) ["""Bind 2 datasets' columns (cbind)"""]  

  * [bind\_X\_and\_target\_y.py](./bind_X_and_target_y.py) ["""Bind dataset and target from another dataset (cbind)"""]  

  * [bind\_n\_datasets.py](./bind_n_datasets.py) ["""Bind multiple datasets by rows (union)"""]  

  * [bootstrap\_time\_series.py](./bootstrap_time_series.py) ["""Bootstrap time series data (bagging and adding) - experimental"""]  

  * [cast\_columns\_to\_numeric.py](./cast_columns_to_numeric.py) ["""Cast columns to numeric values. Columns may have small fraction of non-numeric values"""]  

  * [compute\_rowwise\_stats\_by\_column\_groups.py](./compute_rowwise_stats_by_column_groups.py) ["""Compute rowwise aggregates"""]  

  * [compute\_shift\_diff\_per\_column.py](./compute_shift_diff_per_column.py) ["""Compute shift differences between consecutive rows"""]  

  * [compute\_stats\_by\_groups\_per\_column.py](./compute_stats_by_groups_per_column.py) ["""Compute new features based on aggregates, e.g. distance from mean"""]  

  * [concat\_columns.py](./concat_columns.py) ["""Concatenate columns"""]  

  * [create\_time\_interval\_partition.py](./create_time_interval_partition.py) ["""Create dataset partition based on time interval"""]  

  * [delete\_columns.py](./delete_columns.py) ["""Delete columns based on regex pattern"""]  

  * [delete\_rows.py](./delete_rows.py) ["""Delete rows based on condition(s)"""]  

  * [drop\_duplicates.py](./drop_duplicates.py) ["""Remove duplicate rows"""]  

  * [extract\_header\_data.py](./extract_header_data.py) ["""Extract header (top level) data and drop the rest"""]  

  * [fill\_ts.py](./fill_ts.py) ["""Fill in missing time series rows based on time groups"""]  

  * [filter\_columns\_by\_types.py](./filter_columns_by_types.py) ["""Filter dataset columns by type"""]  

  * [find\_mli\_rowids.py](./find_mli_rowids.py) ["""Get interesting RowIDs to search for in MLI"""]  

  * [impute\_X.py](./impute_X.py) ["""Impute missing values"""]  

  * [insert\_unique\_id.py](./insert_unique_id.py) ["""Add unique row id to a dataset"""]  

  * [join\_X\_left\_outer\_Y.py](./join_X_left_outer_Y.py) ["""Join two datasets"""]  

  * [map\_target\_to\_binary\_outcome.py](./map_target_to_binary_outcome.py) ["""Add new target column with binary label derived from multi-nominal target based on pre-defined rule"""]  

  * [melt\_X.py](./melt_X.py) ["""Melt (unpivot) dataset"""]  

  * [melt\_to\_time\_series.py](./melt_to_time_series.py) ["""Melt (unpivot) time series in wide format to H2O standard long time series format"""]  

  * [parse\_string\_to\_datetime.py](./parse_string_to_datetime.py) ["""Parse string column and convert to date time type"""]  

  * [pivot\_X.py](./pivot_X.py) ["""Pivot dataset"""]  

  * [rename\_column\_names.py](./rename_column_names.py) [""" Rename column name(s) in the dataset"""]  

  * [sample\_X.py](./sample_X.py) [""" Randomly sample rows from dataset"""]  

  * [shift\_time\_series.py](./shift_time_series.py) ["""Manipulate time series values based on time series characteristics"""]  

  * [split\_and\_transpose\_string.py](./split_and_transpose_string.py) ["""Split character value containing list into multiple columns"""]  

  * [split\_by\_datetime.py](./split_by_datetime.py) ["""Split dataset into 2 partitions based on date"""]  

  * [split\_by\_time\_horizon.py](./split_by_time_horizon.py) ["""Split dataset into 2 partitions based on time horizon of test set"""]  

  * [split\_dataset\_by\_partition\_column.py](./split_dataset_by_partition_column.py) ["""Split dataset by partition column - will result in as many partitions as there are values in the partition column"""]  

  * [transform\_features.py](./transform_features.py) ["""Transform dataset features"""]  

