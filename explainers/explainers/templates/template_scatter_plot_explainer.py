"""Scatter plot explainer template which can be used to create explainer with global and local explanations."""

import json

from h2oaicore.mli.oss.byor.core.explainers import CustomExplainer
from h2oaicore.mli.oss.byor.core.explanations import (
    CustomExplanation,
    GlobalScatterPlotExplanation,
)
from h2oaicore.mli.oss.byor.core.representations import (
    GlobalScatterPlotJSonFormat,
)
from h2oaicore.mli.oss.commons import ExplainerModel


class TemplateScatterPlotExplainer(CustomExplainer):
    """Scatter plot explainer template.

    Use this template to create explainer with global and local explanations.

    """

    _display_name = "Template ScatterPlot explainer"
    _description = (
        "Scatter plot explainer template which can be used to create explainer with "
        "global and local explanations."
    )
    _regression = True
    _binary = True
    _multiclass = False
    _global_explanation = True
    _local_explanation = False
    _explanation_types = [GlobalScatterPlotExplanation]
    _keywords = [CustomExplainer.KEYWORD_TEMPLATE]

    def setup(self, model: ExplainerModel, persistence, **kwargs):
        CustomExplainer.setup(
            self, model=model, persistence=persistence, **kwargs
        )

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Explainer returns result mock WITHOUT computation."""

        # explanations list
        explanations = list()

        # global explanation
        explanations.append(self._explain_global_scatter())

        return explanations

    def _explain_global_scatter(self):
        global_explanation = GlobalScatterPlotExplanation(
            explainer=self,
            display_name="Template Scatter Plot",
            display_category=CustomExplanation.DISPLAY_CAT_EXAMPLE,
        )

        #
        # JSon explanation representation formed by multiple files
        #
        json_representation = GlobalScatterPlotJSonFormat(
            explanation=global_explanation,
            json_data=json.dumps(TemplateScatterPlotExplainer.JSON_FORMAT_IDX),
        )
        # add more format files: per-feature, per-class (saved as added to format)
        # (feature and class names MUST fit names from index file ^)
        for clazz in TemplateScatterPlotExplainer.MOCK_CLASSES:
            json_representation.add_data(
                # IMPROVE: tweak values for every class
                format_data=json.dumps(
                    TemplateScatterPlotExplainer.JSON_FORMAT_F_C
                ),
                # filename must fit the name from index file ^
                file_name=f"scatter_{clazz}.json",
            )

        return global_explanation

    #
    # JSon scatter plot mock
    #

    MOCK_CLASSES = ["class_A", "class_B", "class_C"]

    # scatter plot
    JSON_FORMAT_IDX: dict = {
        "files": {
            "class_A": "scatter_class_A.json",
            "class_B": "scatter_class_B.json",
            "class_C": "scatter_class_C.json",
        },
        "total_rows": 20,
        "metrics": [{"R2": 0.96}, {"RMSE": 0.03}],
        "documentation": _description,
    }

    # scatter plot: feature-?, class-?
    JSON_FORMAT_F_C: dict = {
        "bias": 0.15,
        "data": [
            {
                "rowId": 1,
                "responseVariable": 25,
                "limePred": 20,
                "modelPred": 30,
                "actual": 40,
            },
            {
                "rowId": 2,
                "responseVariable": 33,
                "limePred": 15,
                "modelPred": 35,
                "actual": 25,
            },
            {
                "rowId": 3,
                "responseVariable": 35,
                "limePred": 50,
                "modelPred": 30,
                "actual": 40,
            },
            {
                "rowId": 4,
                "responseVariable": 70,
                "limePred": 100,
                "modelPred": 80,
                "actual": 90,
            },
            {
                "rowId": 5,
                "responseVariable": 65,
                "limePred": 80,
                "modelPred": 70,
                "actual": 60,
            },
            {
                "rowId": 6,
                "responseVariable": 50,
                "limePred": 70,
                "modelPred": 75,
                "actual": 65,
            },
        ],
    }
