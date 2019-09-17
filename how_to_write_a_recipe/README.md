# How to write a Transformer Recipe for DriverlessAI?
#### *Ashrith Barthur*

## What is a transformer recipe? 
A transformer (or feature) recipe is a collection of programmatic steps, the same steps that a data scientist would write as code to build a column transformation.  The recipe makes it possible to engineer the transformer in training and in production.
The transformer recipe, and recipes in general, provides a data scientist the power to enhance the strengths of DriverlessAI with custom recipes. These custom recipes would bring in nuanced knowledge about certain domains - i.e. financial crimes, cybersecurity, anomaly detection. etc. It also provides the ability to extend DriverlessAI to solve custom solutions for time-series. 

## How to write a simple DAI recipe? 
The structure of a recipe that works with DriverlessAI is quite straight forward.

1. DriverlessAI provides a `CustomTransformer` Base class that needs to be extended for one to write a recipe. The `CustomTransformer` class provides one the ability to add a customized transformation function. In the following example we are going to create a transformer that will transform a column with the `log10` of the same column. The new column, which is transformed by `log10` will be returned to DriverlessAI as a new column that will be used for modeling. 

```{python eval=FALSE}
class ExampleLogTransformer(CustomTransformer):

```
The `ExampleLogTransformer` is the class name of the transformer that is being newly created. And in the parenthesis the `CustomTransformer` is being extended. 

2. In the next step, one needs to populate the type of problem the custom transformer is solving:
   a. Are you solving a regression problem? 
   b. Are you solving a classification problem that is binary?
   c. Are you solving a classification problem that is multiclass? 
   
Depending on what kind of outcome the custom transformer is solving, each one of the above needs to be enabled or disabled. And the following example will show you how this can be done

```{python eval=FALSE}
class ExampleLogTransformer(CustomTransformer):
	_regression = True
	_binary = True
	_multiclass = True
```
In the above example we are building a `log10` transformer, and this transformer is application, for a regression, binary, or a multiclass problem. Therefore we set all of those as `True`.


3. In the next step, we tackle four more settings of a transformer. They are as follows:
   a. Output Type - What is the output type of this transformer?
   b. Reproducibility - Is this a reproducible transformer? Meaning is this transformer deterministic, and deterministic if you can set the seed?
   c. Model inclusion/exclusion  - Here we describe the type of modeling that uniquely fits, or does not fit the transformer, respectively. 
   4. Custom package requirements - Does this transformer require any custom packages. 
      

```{python eval=FALSE}
class ExampleLogTransformer(CustomTransformer):
	_regression = True
	_binary = True
	_multiclass = True
	_numeric_output = True
	_is_reproducible = True
	_excluded_model_classes = ['tensorflow']
	_modules_needed_by_name = ["custom_package==1.0.0"]
```
In the above example we have set the `_numeric_output` to be `True` as our output is numeric. We have set the `_is_reproducible` to be `True` advicing DriverlessAI that in case the user asks for a reproducible model then this model is actually capable of producing a reproducible result. As an example, we have excluded `tensorflow` using `_excluded_model_classes`. Now, in case, you would want the transformer to only run on a specific kind of model, example - `catboost`, then you can use `_included_model_classes=['CatBoostModel']` instead of `_excluded_model_classes`. Merely, as an example we have also included `custom_package` version `1.0.0` as a package required for this transformation. 

4. In the following section we will discussion about DriverlessAI's ability to check the custom recipe. When the following function is enabled DriverlessAI has the ability to check the workings of the transformer using a synthetic dataset. If this is disabled then DriverlessAI will ingest the recipe but ignore the check. 

```{python eval=FALSE}
class ExampleLogTransformer(CustomTransformer):
	_regression = True
	_binary = True
	_multiclass = True
	_numeric_output = True
	_is_reproducible = True
	_excluded_model_classes = ['tensorflow']
	_modules_needed_by_name = ["custom_package==1.0.0"]

	@staticmethod
	def do_acceptance_test():
	return True
```
In this example we enable the acceptance test by returning `True` for the `do_acceptance_test` function

5. In the following example we set the parameters for the type of column that we require as input, the minimum and the maximum number of columns that we need to be able to provide an output, along with the relative importance of the transformer. 

The column type  or `col_type` can take nine different column data types, and they are as follows:

	a. "all"         - all column types
	b. "any"         - any column types
	c. "numeric"     - numeric int/float column
	d. "categorical" - string/int/float column considered a categorical for feature engineering
	e. "numcat"      - allow both numeric or categorical
	f. "datetime"    - string or int column with raw datetime such as '%Y/%m/%d %H:%M:%S' or '%Y%m%d%H%M'
	g. "date"        - string or int column with raw date such as '%Y/%m/%d' or '%Y%m%d'
	h. "text"        - string column containing text (and hence not treated as categorical)
	i. "time_column" - the time column specified at the start of the experiment (unmodified)

Please note that if `col_type` is set to `col_type=all` then all the columns in the dataframe are provided to this transformer, no selection of columns will occur. 

The `min_cols` and `max_cols` either take numbers/integers or take string parameters as `all` and `any`. The `all` and `any` should coincide with the same `col_type`, respectively. 

The `relative_importance` takes a positive value. If this value is more than `1` then the transformer is likely to be used more often than other transformers in the specific experiment. If it less than `1` then it is less likely to be used than other transformers in the specific experiment. If it is set to `1` then it is equally likely to be used as other transformers in the specific experiment, provided other transformers are also set to relative importance `1`.
i , which will over, or under representation. Default value is `1`, value greater than `1` is over representation and under `1` is under representation. 

```{python eval=FALSE}
class ExampleLogTransformer(CustomTransformer):
	_regression = True
	_binary = True
	_multiclass = True
	_numeric_output = True
	_is_reproducible = True
	_excluded_model_classes = ['tensorflow']
	_modules_needed_by_name = ["custom_package==1.0.0"]

	@staticmethod
	def do_acceptance_test():
	return True

	@staticmethod
	def get_default_properties():
	return dict(col_type = "numeric", min_cols = 1, max_cols = 1, relative_importance = 1)
```

In the above example, as we are dealing with a numeric column (recall, that we are calculating the log10 of a given column) we set the `col_type` to `numeric`. We set the `min_cols` and `max_cols` to `1` as we need only one column, and the `relative_importance` to `1`.

6. The custom transformer function has two fundamental functions that are required to make a transformer. They are:
   a. `fit_transform` This function is used to fit the transformation on the training dataset, and returns the output column. 
   b. `transform` This function is used to transform the testing or production dataset, and is always applied after the `fit_transform`


```{python eval=FALSE}
class ExampleLogTransformer(CustomTransformer):
	_regression = True
	_binary = True
	_multiclass = True
	_numeric_output = True
	_is_reproducible = True
	_excluded_model_classes = ['tensorflow']
	_modules_needed_by_name = ["custom_package==1.0.0"]

	@staticmethod
	def do_acceptance_test():
	return True

	@staticmethod
	def get_default_properties():
	return dict(col_type = "numeric", min_cols = 1, max_cols = 1, relative_importance = 1)

	def fit_transform(self, X: dt.Frame, y: np.array = None):
		X_pandas = X.to_pandas()
		X_p_log = np.log10(X_pandas)
		return X_p_log

	def transform(self, X: dt.Frame):
		X_pandas = X.to_pandas()
		X_p_log = np.log10(X_pandas)
		return X_p_log
```
In the above example, we compose the `fit_transform` and `transform` for training and testing data, respectively. In the `fit_transform` the response variable `y` is available. Here our dataframe is named `X`. Now `X` will be transformed to pandas frame by using the `to_pandas()` function. Further, a `log10` of the column will be applied and returned. The `to_pandas()` function is described here for ease of understanding. 

```{python eval=FALSE}
from h2oaicore.systemutils import segfault, loggerinfo, main_logger
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
import logging
```
We, finally add the required library to the top of the `.py` file. The primary library required is `CustomTransformer`. `loggerinfo` and `main_logger` for house keeping. `datatable` and `pandas` for data handling. 

7. This code is to be stored as a python code file - `example_transform.py`
8. To ingest this code, one needs to first need to add dataset to be modeled upon into DriverlessAI. 
9. After ingestion, `predict` is chosen by rightclicking. Following this a `target` or `response` variable is set.  
10. Next, the `Expert Settings` is chosen, following the recipes, and this - `example_transform.py` is ingested.  
11. Next the transform is chosen under Recipes option and the experiment is started.    
