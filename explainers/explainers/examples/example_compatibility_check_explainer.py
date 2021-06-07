# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from typing import Optional

from h2oaicore.messages import CommonExplainerParameters
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomDaiExplainer,
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation


class ExampleCompatibilityCheckExplainer(CustomExplainer, CustomDaiExplainer):

    _display_name = "Example Compatibility Check Explainer"
    _description = "This is explainer with compatibility check example."
    _regression = True
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)

    def check_compatibility(
        self,
        params: Optional[CommonExplainerParameters] = None,
        **explainer_params,
    ) -> bool:
        CustomExplainer.check_compatibility(self, params, **explainer_params)
        CustomDaiExplainer.check_compatibility(self, params, **explainer_params)

        # explainer can explain only dataset with less than 1M rows (without sampling)
        if self.dataset_entity.row_count > 1_000_000:
            # not supported
            return False
        return True

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]
