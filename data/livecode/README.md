# Live Code Templates for H2O Driverless AI Datasets

## About Driverless AI and BYOR 
See main [README](https://github.com/h2oai/driverlessai-recipes/README.md)

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
[Go to Recipes for Driverless 1.7.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.7.0)
 [1.7.1](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.7.1)
 [1.8.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.0)
 [1.8.1](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.1)
 [1.8.2](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.2)
 [1.8.3](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.3)
 [1.8.4](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.4)
 [1.8.5](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.5)
 [1.8.6](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.6)
 [1.8.7](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.7)
 [1.8.8](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.8)
 [1.9.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.9.0)
### Count: 34
  * [add\_columns\_with\_rand\_values.py](./add_columns_with_rand_values.py) [# Add one or more columns containing random integer values]  

  * [aggregate\_and\_augment\_one\_to\_many.py](./aggregate_and_augment_one_to_many.py) [# Template for augmenting data based on one-to-many relationship between datasets X (e.g. header level) ]  

  * [aggregate\_columns\_with\_groupby.py](./aggregate_columns_with_groupby.py) [# Compute aggregates with per-column expressions (means and sums in this example)]  

  * [balance\_data.py](./balance_data.py) [# Create a sampled dataset for imbalanced use cases - probably not for modeling but]  

  * [bind\_2\_datasets.py](./bind_2_datasets.py) [# Livecode for binding 2 datasets' rows (rbind). Datasets should have the same]  

  * [bind\_X\_and\_Y.py](./bind_X_and_Y.py) [# Template for binding columns from 2 datasets with the same number of rows.]  

  * [bind\_X\_and\_target\_y.py](./bind_X_and_target_y.py) [# Template for binding dataset and target from another dataset with the same number of rows,]  

  * [bind\_n\_datasets.py](./bind_n_datasets.py) [# Livecode for binding multiple datasets' rows (rbind). Datasets should have the same]  

  * [cast\_columns\_to\_numeric.py](./cast_columns_to_numeric.py) [# Cast columns with mostly numeric values to new numeric columns.]  

  * [compute\_shift\_diff\_per\_column.py](./compute_shift_diff_per_column.py) [# Compute per-column difference between current and previous (shift)]  

  * [compute\_stats\_by\_groups\_per\_column.py](./compute_stats_by_groups_per_column.py) [# Compute per-column expressions (signed distance from the mean in this example) ]  

  * [create\_time\_interval\_partition.py](./create_time_interval_partition.py) [# Extract single partition based on time interval]  

  * [delete\_columns.py](./delete_columns.py) [# Delete columns with the names matching regular expression pattern.]  

  * [delete\_rows.py](./delete_rows.py) [# Delete rows based on certain condition.]  

  * [drop\_duplicates.py](./drop_duplicates.py) [# Remove duplicate rows by grouping the same rows,]  

  * [extract\_header\_data.py](./extract_header_data.py) [# Extract header data from detailed (line item level) dataset by filtering top level ]  

  * [fill\_ts.py](./fill_ts.py) [# Add any missing Group by Date records and fill with a default value -]  

  * [filter\_columns\_by\_types.py](./filter_columns_by_types.py) [# Filter only columns of certain types. Beware that column order]  

  * [find\_mli\_rowids.py](./find_mli_rowids.py) [# Get interesting RowIDs to search for in MLI]  

  * [impute\_X.py](./impute_X.py) [# Live code recipe for imputing all missing values]  

  * [insert\_unique\_id.py](./insert_unique_id.py) [# Livecode to add (insert) new column containing unique row]  

  * [join\_X\_left\_outer\_Y.py](./join_X_left_outer_Y.py) [# Livecode for joining 2 datasets, e.g.]  

  * [map\_target\_to\_binary\_outcome.py](./map_target_to_binary_outcome.py) [# Maps multi-nominal target (outcome) to binomial target column by]  

  * [melt\_X.py](./melt_X.py) [# Change dataset format from wide to long using melt function]  

  * [melt\_to\_time\_series.py](./melt_to_time_series.py) [# Melt time series in wide format (single row) into long format supported]  

  * [parse\_string\_to\_datetime.py](./parse_string_to_datetime.py) [# Parse and convert string column to date.]  

  * [pivot\_X.py](./pivot_X.py) [# Change dataset format from long to wide using pivot function]  

  * [rename\_column\_names.py](./rename_column_names.py) [# Rename column name(s) in the dataset]  

  * [sample\_X.py](./sample_X.py) [# Random sample of rows from X]  

  * [split\_and\_transpose\_string.py](./split_and_transpose_string.py) [# Template to parse and split a character column using pandas str.split, ]  

  * [split\_by\_datetime.py](./split_by_datetime.py) [# Split dataset into two partitions by time given]  

  * [split\_by\_time\_horizon.py](./split_by_time_horizon.py) [# Split dataset into two partitions by time given]  

  * [split\_dataset\_by\_partition\_column.py](./split_dataset_by_partition_column.py) [# Split dataset by partition id (column): results in as many partitions (datasets)]  

  * [transform\_features.py](./transform_features.py) [# Map and create new features by adding new columns or with in-place update.]  

