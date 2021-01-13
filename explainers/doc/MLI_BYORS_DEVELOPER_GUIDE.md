# MLI BYORs Developer Guide
MLI BYORs developer guide.
## Best Practices

### Versioning
There is no Driverless AI BYOR recipe versioning API. To deploy new recipe **version** (while you keep the previous one) change:

* recipe **class name**
* recipe **display name**

This will create a new explainer with new name like _"DT surrogate explainer v2.1"_.

### Security
* Recipes are meant to be built by people you trust and each recipe should be code-reviewed before going to production.
* Assume that a user with access to Driverless AI has access to the data inside that instance.
  * Apart from securing access to the instance via private networks, various methods of [authentication](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/authentication.html) are possible. Local authentication provides the most control over which users have access to Driverless AI.
  * Unless the `config.toml` setting `enable_dataset_downloading=false` is set, an authenticated user can download all imported datasets as .csv via direct APIs.
* When recipes are enabled (`enable_custom_recipes=true`, the default), be aware that:
  * The code for the recipes runs as the same native Linux user that runs the Driverless AI application.
    * Recipes have explicit access to all data passing through the transformer/model/scorer API
    * Recipes have implicit access to system resources such as disk, memory, CPUs, GPUs, network, etc.
  * A H2O-3 Java process is started in the background, for use by all recipes using H2O-3. Anyone with access to the Driverless AI instance can browse the file system, see models and data through the H2O-3 interface.

* Best ways to control access to Driverless AI and custom recipes:
  * Control access to the Driverless AI instance
  * Use local authentication to specify exactly which users are allowed to access Driverless AI
  * Run Driverless AI in a Docker container, as a certain user, with only certain ports exposed, and only certain mount points mapped
  * To disable all recipes: Set `enable_custom_recipes=false` in the config.toml, or add the environment variable `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES=0` at startup of Driverless AI. This will disable all custom transformers, models and scorers.
  * To disable new recipes: To keep all previously uploaded recipes enabled and disable the upload of any new recipes, set `enable_custom_recipes_upload=false` or `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES_UPLOAD=0` at startup of Driverless AI.
### Safety
* Driverless AI automatically performs basic acceptance tests for all custom recipes unless disabled.
* More information in the FAQ.
### Performance
* Use fast and efficient data manipulation tools like `data.table`, `sklearn`, `numpy` or `pandas` instead of Python lists, for-loops etc.
* Use disk sparingly, delete temporary files as soon as possible.
* Use memory sparingly, delete objects when no longer needed.
## Python Client API reference
```
ExplanationDescriptor
  explanation_type str
  name str
  category str
  scope str
  has_local str
  formats str[]

ExplainerDescriptor
  id str
  name str
  model_types str[]
  can_explain str[]
  explanation_scopes str[]
  explanations ExplanationDescriptor[]
  keywords str[]

ExplainerRunJob
  progress float
  status int
  error str
  message str
  entity ExplainerDescriptor
  created float
  duration int
  child_explainers_job_keys str[]

ExplainersRunJob
  explainer_job_keys str[]
  mli_key str
  created float
  duration int
  status int
  progress float

CommonExplainerParameters
  target_col str
  weight_col str
  prediction_col str # no model explanation
  drop_cols str[]
  sample_num_rows int # >0 to sample, -1 to skip sampling

CommonDaiExplainerParameters
  common_params CommonExplainerParameters
  model ModelReference
  dataset DatasetReference
  validset DatasetReference
  testset DatasetReference
  use_raw_features bool
  config_overrides str
  sequential_execution bool
  debug_model_errors bool
  debug_model_errors_class str

Explainer
  explainer_id str # explainer ID
  explainer_params str # explainer parameters as JSon string

ExplainersRunSummary
  common_params CommonExplainerParameters
  explainers Explainer[]
  explainer_run_jobs ExplainerRunJob[]

list_explainers ExplainerDescriptor[]
  experiment_types str[]
  explanation_scopes str[]
  dai_model_key str
  keywords str[]
  explainer_filter FilterEntry[]

list_explainable_models ListModelQueryResponse
  explainer_id str
  offset int
  size int

get_explainer ExplainerDescriptor
  explainer_id str

run_explainers ExplainersRunJob
  explainers Explainer[] # explainers to run
  params CommonDaiExplainerParameters # common DAI explainer run parameters

get_explainer_run_job ExplainerRunJob
  explainer_job_key str

abort_explainer_run_jobs void
  explainer_job_keys str[]

get_explainer_run_log_url_path str
  mli_key str
  explainer_job_key str

list_explainer_results ExplainerDescriptor
  explainer_job_key str

get_explainer_result_url_path str
  mli_key str
  explainer_job_key str
  explanation_type str
  explanation_format str

get_explainer_snapshot_url_path str*
  mli_key str
  explainer_job_key str

FilterEntry
  filter_by str
  value str

get_explainer_result str
  mli_key str
  explainer_job_key str
  explanation_type str
  explanation_format str
  page_offset int
  page_size int
  result_format str
  explanation_filter FilterEntry[]

get_explainer_local_result str
  mli_key str
  explainer_job_key str
  explanation_type str
  explanation_format str
  id_column_name str
  id_column_value str
  page_offset int
  page_size int
  result_format str
  explanation_filter FilterEntry[]
```
