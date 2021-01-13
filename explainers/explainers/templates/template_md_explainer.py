# Copyright 2017-2020 H2O.ai, Inc. All rights reserved.
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from h2oaicore.mli.oss.byor.core.explainers import CustomExplainer
from h2oaicore.mli.oss.byor.core.explanations import (
    CustomExplanation,
    ReportExplanation,
)
from h2oaicore.mli.oss.byor.core.representations import MarkdownFormat
from h2oaicore.mli.oss.commons import ExplainerModel


class TemplateMarkdownExplainer(CustomExplainer):
    """Markdown report with raster image chart explainer template.

    Use this template to create explainer with global report explanations.

    """

    _display_name = "Template Markdown explainer"
    _description = (
        "Markdown report with raster image chart explainer template which can be used "
        "to create explainer with global report explanations."
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
            display_name="Template Markdown report",
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
        TemplateMarkdownExplainer._create_report_image(work_img_path)
        # save report
        report_path = self.persistence.get_explainer_working_file("report.md")
        with open(report_path, mode="w") as file:
            file.write(MARKDOWN_TEMPLATE.format(img_file_name))

        return report_path, [work_img_path]

    @staticmethod
    def _create_report_image(img_path: str):
        ts = pd.Series(
            np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000)
        )
        ts = ts.cumsum()
        df = pd.DataFrame(
            np.random.randn(1000, 4), index=ts.index, columns=list("ABCD")
        )
        df = df.cumsum()
        plt.figure()
        df.plot()
        plt.savefig(img_path, dpi=300)


#
# Markdown report
#

MARKDOWN_TEMPLATE: str = """# Example Report
This is an example of **Markdown report** which can be created by explainer.

![image](./{})

"""
