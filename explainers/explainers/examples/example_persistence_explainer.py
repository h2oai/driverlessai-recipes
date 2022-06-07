# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation


class ExamplePersistenceExplainer(CustomExplainer):
    _display_name = "Example Persistence Explainer"
    _description = (
        "This is explainer example which demonstrates how to use persistence object"
        "in order to access explainer file system (sandbox) - working, explanations "
        "and MLI directories."
    )
    _regression = True
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)

    def setup(self, model, persistence, **kwargs):
        CustomExplainer.setup(self, model, persistence, **kwargs)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        # use self.persistence object to get file system paths
        self.logger.info(f"Explainer MLI dir: {self.persistence.base_dir}")
        self.logger.info(
            f"Explainer dir: {self.persistence.get_explainer_dir()}"
        )

        # save 1st row of dataset to work directory and prepare work directory archive
        df_head = X[:1, :]
        df_head.to_csv(
            self.persistence.get_explainer_working_file("dataset_head.csv")
        )

        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]
