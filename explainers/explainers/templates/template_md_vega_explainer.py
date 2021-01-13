 """Markdown report with Vega chart explainer template which can be used to create explainer which creates global report explanations.""""
 
from h2oaicore.mli.oss.byor.core.explainers import CustomExplainer
from h2oaicore.mli.oss.byor.core.explanations import (
    CustomExplanation,
    ReportExplanation,
)
from h2oaicore.mli.oss.byor.core.representations import MarkdownFormat
from h2oaicore.mli.oss.commons import ExplainerModel


class TemplateMarkdownVegaExplainer(CustomExplainer):
    """Markdown report with Vega chart explainer template.

    Use this template to create explainer with global report explanations.

    """

    _display_name = "Template Markdown with Vega explainer"
    _description = (
        "Markdown report with Vega chart explainer template which can be used to "
        "create explainer which creates global report explanations."
    )
    _regression = True
    _binary = True
    _multiclass = False
    _global_explanation = True
    _explanation_types = [ReportExplanation]
    _keywords = [CustomExplainer.KEYWORD_TEMPLATE]

    def setup(self, model: ExplainerModel, persistence, **kwargs):
        CustomExplainer.setup(
            self, model=model, persistence=persistence, **kwargs
        )

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Create global and local (pre-computed/cached) explanations.

        Template explainer returns MOCK explanation data - replace mock data
        preparation with actual computation to create real explainer.

        """
        # explanations list
        explanations = list()

        # global explanation
        explanations.append(self.explain_global_markdown(self.persistence))

        return explanations

    def explain_global_markdown(self, persistence):
        global_explanation = ReportExplanation(
            explainer=self,
            display_name="Template Markdown report with Vega",
            display_category=CustomExplanation.DISPLAY_CAT_EXAMPLE,
        )

        #
        # MD explanation representation formed by multiple files
        #
        report_path = persistence.get_explainer_working_file("report.md")
        with open(report_path, mode="w") as file:
            # Vega code must be quoted in case Python string is formatted
            file.write(TemplateMarkdownVegaExplainer.MARKDOWN_TEXT)

        global_explanation.add_format(
            MarkdownFormat(
                explanation=global_explanation, format_file=report_path
            )
        )

        return global_explanation

    #
    # Markdown
    #

    MARKDOWN_TEXT: str = """# Example Report
Vega heatmap:

```vega
{
  "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
  "data": {
    "values": [
      {"actual": "A", "predicted": "A", "count": 13},
      {"actual": "A", "predicted": "B", "count": 0},
      {"actual": "A", "predicted": "C", "count": 0},
      {"actual": "B", "predicted": "A", "count": 0},
      {"actual": "B", "predicted": "B", "count": 10},
      {"actual": "B", "predicted": "C", "count": 6},
      {"actual": "C", "predicted": "A", "count": 0},
      {"actual": "C", "predicted": "B", "count": 0},
      {"actual": "C", "predicted": "C", "count": 9}
    ]
  },
  "selection": {
    "highlight": {"type": "single"}
  },
  "mark": {"type": "rect", "strokeWidth": 2},
  "encoding": {
    "y": {
      "field": "actual",
      "type": "nominal"
    },
    "x": {
      "field": "predicted",
      "type": "nominal"
    },
    "fill": {
      "field": "count",
      "type": "quantitative"
    },
    "stroke": {
      "condition": {"test": {"and": [{"selection": "highlight"},
      "length(data('highlight_store'))"]}, "value": "black"},
      "value": null
    },
    "opacity": {
      "condition": {"selection": "highlight", "value": 1},
      "value": 0.5
    },
    "order": {"condition": {"selection": "highlight", "value": 1}, "value": 0}
  },
  "config": {
    "scale": {
      "bandPaddingInner": 0,
      "bandPaddingOuter": 0
    },
    "view": {"step": 40},
    "range": {
      "ramp": {
        "scheme": "yellowgreenblue"
      }
    },
    "axis": {
      "domain": false
    }
  }
}
```
<br/>
Vega column chart:

```vega
{
  "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
  "description": "A simple bar chart with embedded data.",
  "data": {
    "values": [
      {"a": "A", "b": 28}, {"a": "B", "b": 55}, {"a": "C", "b": 43},
      {"a": "D", "b": 91}, {"a": "E", "b": 81}, {"a": "F", "b": 53},
      {"a": "G", "b": 19}, {"a": "H", "b": 87}, {"a": "I", "b": 52}
    ]
  },
  "mark": "bar",
  "encoding": {
    "x": {"field": "a", "type": "nominal", "axis": {"labelAngle": 0}},
    "y": {"field": "b", "type": "quantitative"}
  }
}
```

"""
