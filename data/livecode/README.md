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
### Count: 30
  * [add\_columns\_with\_rand\_values.py](./add_columns_with_rand_values.py) [ Add one or more columns containing random integer values  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       col_count: int - number of random columns to add  

      

       random_column_names: List[str] - names of the columns  

      

       min_max_values: tuple - the reange (minimum and maximum) to pick random integers from  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [aggregate\_columns\_with\_groupby.py](./aggregate_columns_with_groupby.py) [ Compute aggregates with per-column expressions (means and sums in this example)  

      

     for numeric (int, float) columns by groups.  

      

     New frame contains computed aggregates of the columns and group by columns.  

      

     see: compute_stats_by_groups_per_column.py and  

      

          https://stackoverflow.com/questions/62974899/updating-or-adding-multiple-columns-with-pydatatable-in-style-of-r-datables-sd  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

       mean_columns: list of str - columns to compute means on (change to the aggregates and columns of your choice)  

      

       sum_columns: list of str - columns to compute sums on (change to the aggregates and columns of your choice)  

      

     Parameters:  

      

       group_by_cols: list of column names - group columns to aggregate by  

      

     Output:  

      

       dataset with computed aggregates and groups  

      

    ]  

  * [balance\_data.py](./balance_data.py) [ Create a sampled dataset for imbalanced use cases - probably not for modeling but  

      

     can be nice to better see trends in MLI PDP plots  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       target_col: str - usually target column to use when balancing data  

      

       times: int - how much to downsample majority class: in number of times size of minority class  

      

       random_seed: int - random seed to control for reproducibility  

      

     Output:  

      

       dataset with downsampled majority class  

      

    ]  

  * [bind\_2\_datasets.py](./bind_2_datasets.py) [ Livecode for binding 2 datasets' rows (rbind). Datasets should have the same  

      

     columnar structure, e.g. train dataset and test dataset (with target present).  

      

     For more details see docs on datatable's Frame.rbind() here:  

      

     https://datatable.readthedocs.io/en/latest/api/frame.html#datatable.Frame.rbind  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

       X2_name: datatable - dataset to bind with  

      

     Parameters:  

      

       None  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [bind\_X\_and\_Y.py](./bind_X_and_Y.py) [ Template for binding columns from 2 datasets with the same number of rows.  

      

     Recipe won't perform any joins/mapping but rather stitch 2 datasets together into wider dataset with  

      

     the same number of rows and columns from both.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

       Y_name: string - dataset location to bind with  

      

     Parameters:  

      

       None  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [bind\_X\_and\_target\_y.py](./bind_X_and_target_y.py) [ Template for binding dataset and target from another dataset with the same number of rows,  

      

     e.g. one dataset has features and another contains target.  

      

     Recipe won't perform any joins/mapping but rather stitch 2 datasets together into wider dataset with  

      

     the same number of rows and columns from 1st dataset plus target from another.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

       y_name: string - location of the dataset containing target value  

      

     Parameters:  

      

       target_col: string - target name  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [bind\_n\_datasets.py](./bind_n_datasets.py) [ Livecode for binding multiple datasets' rows (rbind). Datasets should have the same  

      

     columnar structure, e.g. each file contains one month of train data.  

      

     For more details see docs on datatable's Frame.rbind() here:  

      

     https://datatable.readthedocs.io/en/latest/api/frame.html#datatable.Frame.rbind  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

       files_to_bind: list of datatables - datasets to bind with  

      

     Parameters:  

      

       None  

      

     Output:  

      

       dataset containing all rows from primary and the list datasets  

      

    ]  

  * [cast\_columns\_to\_numeric.py](./cast_columns_to_numeric.py) [ Cast columns with mostly numeric values to new numeric columns.  

      

     Fraction of non-numeric values (per column) equal or below a threshold defined is allowed before casting.  

      

     Non-numeric values are ignored and mapped to null value.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       columns: list - columns to cast to numeric. If None (default) then all character columns  

      

       threshold_fraction_non_numeric: numeric - threshold for percentage of allowed non-numeric values per column before conversion  

      

       in_place: bool - indicates if casting to numeric takes place in place or by adding new column  

      

       name_suffix: string - suffix to add to the name of new numeric column converted from the original  

      

     Output:  

      

       dataset with added numeric columns derived from character columns  

      

    ]  

  * [compute\_shift\_diff\_per\_column.py](./compute_shift_diff_per_column.py) [ Compute per-column difference between current and previous (shift)  

      

     values for each time series - both by time groups (multiple time  

      

     series) and across covariates (multivariate time series).  

      

     Multiple time series identified by group columns while   

      

     covariates are explicitly assigned in `shift_cols`.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       time_col: date/time/int - time column to order rows before the shift op  

      

       group_by_cols: list of column names - group columns  

      

       shift_cols: list of column names - columns to shift  

      

     Output:  

      

       dataset augmented with shifted columns  

      

    ]  

  * [compute\_stats\_by\_groups\_per\_column.py](./compute_stats_by_groups_per_column.py) [ Compute per-column expressions (signed distance from the mean in this example)   

      

     for all numeric (int, float) columns with stats computed by groups and  

      

     new column added for each original numeric feature.  

      

     see: https://stackoverflow.com/questions/62974899/updating-or-adding-multiple-columns-with-pydatatable-in-style-of-r-datables-sd  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       group_by_cols: list of column names - group columns to compute stats by  

      

     Output:  

      

       dataset augmented with computed statistics  

      

    ]  

  * [create\_time\_interval\_partition.py](./create_time_interval_partition.py) [ Extract single partition based on time interval  

      

     Data is called X and is a DataTable object  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       date_col: date/time/int - time column to order rows  

      

       split_date_min: lower bound of partition interval  

      

       split_date_max: upper bound of partition interval  

      

     Output:  

      

       dataset containing partition interval  

      

    ]  

  * [delete\_columns.py](./delete_columns.py) [ Delete columns with the names matching regular expression pattern.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       col_name_regex: str - regular expression pattern  

      

     Output:  

      

       dataset containing only column names that do not match the pattern  

      

    ]  

  * [delete\_rows.py](./delete_rows.py) [ Delete rows based on certain condition.  

      

     In this case delete rows where certain column contains null values.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       col_name: str - column name  

      

     Output:  

      

       dataset containing only rows with non-null values in designated column  

      

    ]  

  * [drop\_duplicates.py](./drop_duplicates.py) [ Remove duplicate rows by grouping the same rows,  

      

     sorting them and then selecting first (1) or last (-1)  

      

     row from each group  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       sort_cols: date/time/int/str - column(s) to order rows within each group  

      

       key_cols: list of column names - group columns  

      

     Output:  

      

       dataset after removing dups  

      

    ]  

  * [fill\_ts.py](./fill_ts.py) [ Add any missing Group by Date records and fill with a default value -  

      

     additional columns will be null for the default values  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       ts_col: date/time - temporal column  

      

       group_by_cols: list of columns - column(s) to define groups of rows  

      

       target_col: list of column names - group columns  

      

       default_missing_value: - value to fill when missing found  

      

     Output:  

      

       dataset augmented with missing data  

      

    ]  

  * [filter\_columns\_by\_types.py](./filter_columns_by_types.py) [ Filter only columns of certain types. Beware that column order  

      

     changes after filtering. For more details see f-expressions in   

      

     datatable docs:   

      

     https://datatable.readthedocs.io/en/latest/manual/f-expressions.html#f-expressions  

      

     E.g. below all integer and floating-point columns are retained   

      

     while the others are dropped. Because int type is followed by  

      

     float type columns are re-shuffled so all integer columns   

      

     placed first and then float ones.  

      

     For reference various data type filters are listed.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       None explicitly, filtering columns by types inside.  

      

     Output:  

      

       dataset with columns filtered by data types  

      

    ]  

  * [find\_mli\_rowids.py](./find_mli_rowids.py) [ Get interesting RowIDs to search for in MLI  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       target_col: list of column names - group columns  

      

     Output:  

      

       dataset with selected rows and ids  

      

    ]  

  * [impute\_X.py](./impute_X.py) [ Live code recipe for imputing all missing values  

      

     in a dataset  

      

     If you don't want certain data type to be filled just   

      

     change its filler's value to None  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       fill_int: integer - filler for missing integer values  

      

       fill_float: numeric - filler for missing float values  

      

       fill_char:  string - filler for missing string values  

      

       fill_bool: bool - filler for missing logical values  

      

     Output:  

      

       dataset with filled values  

      

    ]  

  * [insert\_unique\_id.py](./insert_unique_id.py) [ Livecode to add (insert) new column containing unique row  

      

     identifier. New dataset will be identical to its source  

      

     plus inserted first column containing unique ids from 0 to N-1  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       column_name: string - new column name to store id values  

      

     Output:  

      

       dataset augmented with id column  

      

    ]  

  * [join\_X\_left\_outer\_Y.py](./join_X_left_outer_Y.py) [ Livecode for joining 2 datasets, e.g.  

      

     one dataset with transactions and another dataset has extended set of features.  

      

     find location of the dataset file by going to DETAILS where it's displayed  

      

     on top under dataset name  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

       Y_name: datatable - dataset to bind with  

      

     Parameters:  

      

       join_key: string - column name to use as a key in join  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [map\_target\_to\_binary\_outcome.py](./map_target_to_binary_outcome.py) [ Maps multi-nominal target (outcome) to binomial target column by  

      

     binding new column to a dataset (new dataset will be created).  

      

     For example, use when working with multi-nominal classifier and want   

      

     to see if binomial model may be preferred or compliment use case.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       target_name: string - target column name  

      

       new_target_name: string - new target column name  

      

       value_to_map_to_true: value - target values that maps to binary positive (true) outcome  

      

       binary_outcomes: tuple - pair of binary outcomes to sue for new target  

      

       drop_old_target: bool - if true then drop old target column  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [melt\_X.py](./melt_X.py) [ Change dataset format from wide to long using melt function  

      

     Identify id columns and value columns to use Pandas melt   

      

     function  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       id_cols: list of columns - columns to use as identifier variables  

      

       value_col_regex: string - regular expression pattern to select value columns  

      

       value_cols: list of columns - columns to unpivot (melt) to use (if regex 'value_col_regex' is None)  

      

       var_name: string - name to use for the 'variable' columns  

      

       value_name: string - name to use for the 'value' column  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [melt\_to\_time\_series.py](./melt_to_time_series.py) [ Melt time series in wide format (single row) into long format supported  

      

     by DAI: a row represents a point in time (lag) so a column represents  

      

     a time series values.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       id_cols: list of columns - columns to use as identifier variables  

      

       time_series_col_name: string - name of time series columns  

      

       time_series_new_name: string - name to use in melted data  

      

       timestamp_col_name: string - column name for time values  

      

     Output:  

      

       dataset with melted time series  

      

    ]  

  * [parse\_string\_to\_datetime.py](./parse_string_to_datetime.py) [ Parse and convert string column to date.  

      

     This example converts string in the format `MMMMYY` to `MMMM-YY-DD`.  

      

     Please adjust code to the format you expect to find in your data.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary data set  

      

     Parameters:  

      

       col_name: str - column containing string to parse as date/time  

      

       date_col_name: str - new column to store parsed date/time  

      

     Output:  

      

       dataset with parsed date  

      

    ]  

  * [pivot\_X.py](./pivot_X.py) [ Change dataset format from long to wide using pivot function  

      

     Identify id columns and value columns to use Pandas pivot   

      

     function  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       id_cols: list of columns - column to use to make new frame’s index  

      

       var_name: string - name to use for the 'variable' columns  

      

       value_name: string - name to use for the 'value' column  

      

     Output:  

      

       dataset containing all rows from both datasets  

      

    ]  

  * [rename\_column\_names.py](./rename_column_names.py) [ Rename column name(s) in the dataset  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       column_rename_map: dictionary - mapping old column names to new ones  

      

    ]  

  * [sample\_X.py](./sample_X.py) [ Random sample of rows from X  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       fraction: float - fraction of rows to sample from 'X' (must be between 0 and 1)  

      

       random_seed: int - random seed to control for reproducibility  

      

    ]  

  * [split\_by\_datetime.py](./split_by_datetime.py) [ Split dataset into two partitions by time given  

      

     date/time value.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       date_col: string - name of temporal column  

      

       split_date: date/time - temporal value to split dataset on  

      

       date_format: string - date format to parse date in pandas, if None then no parsing takes place  

      

    ]  

  * [split\_by\_time\_horizon.py](./split_by_time_horizon.py) [ Split dataset into two partitions by time given  

      

     time horizon (length) of last partition. With this  

      

     approach we simply count number of unique values in temporal  

      

     column and take the N-th from the end to be the border value.  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       date_col: string - name of temporal column  

      

       forecast_len: integer - length of last partition measured in temporal units used in X  

      

    ]  

  * [split\_dataset\_by\_partition\_column.py](./split_dataset_by_partition_column.py) [ Split dataset by partition id (column): results in as many partitions (datasets)  

      

     as there are values in parition column  

      

      

      

     Specification:  

      

     Inputs:  

      

       X: datatable - primary dataset  

      

     Parameters:  

      

       partition_col_name: string - column name identifying which partition row belongs to  

      

       dataset_name_prefix: string - prefix to use in the names for new datasets  

      

       MAX_PARTITIONS: int - maximum number of partition datasets to create  

      

    ]  

