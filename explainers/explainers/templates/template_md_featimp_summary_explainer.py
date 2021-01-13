"""Markdown report with summary feature importance chart explainer template which can be used to create explainer with global report explanations."""

import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from h2oaicore.mli.oss.byor.core.explainers import CustomExplainer
from h2oaicore.mli.oss.byor.core.explanations import (
    CustomExplanation,
    ReportExplanation,
)
from h2oaicore.mli.oss.byor.core.representations import MarkdownFormat
from h2oaicore.mli.oss.byor.library.representations import ScatterFeatImpPlot
from h2oaicore.mli.oss.commons import ExplainerModel


class TemplateMarkdownFeatImpSummaryExplainer(CustomExplainer):
    """Markdown report with summary feature importance chart explainer template.

    Use this template to create explainer with global report explanations.

    """

    _display_name = "Template Markdown feature importance summary explainer"
    _description = (
        "Markdown report with summary feature importance chart explainer template "
        "which can be used to create explainer with global report explanations."
    )
    _regression = True
    _binary = True
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
        explanations.append(self.explain_global_markdown())

        return explanations

    def explain_global_markdown(self):
        global_explanation = ReportExplanation(
            explainer=self,
            display_name="Template Feature Importance Summary Markdown report",
            display_category=CustomExplanation.DISPLAY_CAT_EXAMPLE,
        )

        # CALCULATION: Markdown report with image(s) in work directory
        report_path, images_path = self._create_report()

        # NORMALIZATION: Markdown report to Grammar of MLI format in Driverless AI UI
        global_explanation.add_format(
            MarkdownFormat(
                explanation=global_explanation,
                format_file=report_path,
                extra_format_files=images_path,
            )
        )

        return global_explanation

    def _create_report(self) -> Tuple[str, List[str]]:
        # save image
        img_file_name = "image.png"
        work_img_path = self.persistence.get_explainer_working_file(
            img_file_name
        )
        TemplateMarkdownFeatImpSummaryExplainer._create_report_image(
            work_img_path
        )
        # save report
        report_path = self.persistence.get_explainer_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(MARKDOWN_TEMPLATE.format(img_file_name))

        return report_path, [work_img_path]

    @staticmethod
    def _create_report_image(img_path: str):
        def generate_chart_data(max_features: int = 10):
            data: dict = {}
            for i in range(max_features):
                data[f"feature_{i}"] = [
                    random.uniform(0, 1) for _ in range(100)
                ]
            return data

        contributions = pd.DataFrame(generate_chart_data())
        frame = pd.DataFrame(generate_chart_data())

        plot = ScatterFeatImpPlot.plot(contributions=contributions, frame=frame)

        plot.savefig(fname=img_path)

        plt.savefig(img_path, dpi=300)


#
# Markdown report
#

MARKDOWN_TEMPLATE: str = """# Example Feature Importance Summary Report
This is an example of **Markdown report** which can be created by explainer.

![image](./{})

"""
