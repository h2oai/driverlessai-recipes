# Recipes for Machine Learning Interpretability in Driverless AI

* [What are MLI Custom Recipes?](#what-are-mli-custom-recipes)
* [Best Practices for MLI Recipes](#best-practices-for-mli-recipes)
* [Resources](#resources)
* [MLI Recipes](#mli-recipes)

## What are MLI Custom Recipes?
Driverless AI provides robust interpretability of machine learning models to explain modeling results in a human-readable format. In the Machine Learning Interpetability (MLI) view, Driverless AI employs a host of different techniques and methodologies for interpreting and explaining the results of its models.

The set of MLI techniques and methodologies can be extended with **recipes**. With MLI BYOR, you can use your own recipes in combination with or instead of all built-in recipes. This allows you to further extend MLI **explainers** in addition to out-of-the-box techniques.

Custom explainer recipes can be uploaded into Driverless AI at runtime without having to restart the platform, just like a plugin.
## Best Practices for MLI Recipes
For [security](doc/MLI_BYORS_DEVELOPER_GUIDE.md#security), [safety](doc/MLI_BYORS_DEVELOPER_GUIDE.md#safety) and [performance](doc/MLI_BYORS_DEVELOPER_GUIDE.md#performance) best practices please refer to [developer guide](doc/MLI_BYORS_DEVELOPER_GUIDE.md)
## Resources
* Tutorials: 
    * [Creating Custom Explainer with MLI BYORs](doc/CREATING_CUSTOM_EXPLAINER_WITH_MLI_BYOR.md)
* Notebooks:
    * [MLI BYORs Python client in examples](notebooks/mli-byor.ipynb)
* Documentation:
	* [Developer guide](doc/MLI_BYORS_DEVELOPER_GUIDE.md)
    * [API reference](doc/api)
    * [MLI documentation](https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/interpreting.html)
    
## MLI Recipes
**IID** explainers:

* [Morris sensitivity analysis](explainers/morris_sensitivity_explainer.py)

<!--
**Time series** explainers:

* .
-->

Explainer **templates**:

* [Feature importance explainer template](explainers/templates/template_featimp_explainer.py)
* [PD/ICE explainer template](explainers/templates/template_pd_explainer.py)
* [Decision tree explainer template](explainers/templates/template_dt_explainer.py)
* [Markdown report explainer template](explainers/templates/template_md_explainer.py)
* [Markdown report with Vega charts template](explainers/templates/template_md_vega_explainer.py)
* [Markdown feature importance summary explainer template](explainers/templates/template_md_featimp_summary_explainer.py)
* [Scatter plot explainer template](explainers/templates/template_scatter_plot_explainer.py)

Explainer **examples**:

* [Hello, World! explainer example](explainers/examples/example_hello_world_explainer.py)
* [Filter and score explainer example](explainers/examples/example_score_explainer.py)
* [Dataset and EDA explainer example](explainers/examples/example_eda_explainer.py)
* [Explainer parameters example](explainers/examples/example_params_explainer.py)
* [Explainer logging example](explainers/examples/example_logging_explainer.py)
* [Explainer persistence example](explainers/examples/example_persistence_explainer.py)
* [Explainer compatibility check example](explainers/examples/example_compatibility_check_explainer.py)
* [Custom explanation example](explainers/examples/example_custom_explanation_explainer.py)
* [Driverless AI metadata explainer example](explainers/examples/example_dai_metadata_explainer.py)
