# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomDaiExplainer,
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation


class ExampleMetaAndAttrsExplainer(CustomExplainer, CustomDaiExplainer):

    _display_name = "Example DAI Explainer Metadata and Attributes"
    _description = (
        "This explainer example prints explainer metadata, instance attributes and "
        "setup() method parameters."
    )
    _regression = True
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)

    def setup(self, model, persistence, **e_params):
        CustomExplainer.setup(self, model, persistence, **e_params)
        CustomDaiExplainer.setup(self, **e_params)

        self.logger.info("setup() method parameters:")
        self.logger.info(f"    {e_params}")

        self.logger.info("explainer metadata:")
        self.logger.info(f"    display name: {self._display_name}")
        self.logger.info(f"    description: {self._description}")
        self.logger.info(f"    keywords: {self._keywords}")
        self.logger.info(f"    IID: {self._iid}")
        self.logger.info(f"    TS: {self._time_series}")
        self.logger.info(f"    image: {self._image}")
        self.logger.info(f"    regression: {self._regression}")
        self.logger.info(f"    binomial: {self._binary}")
        self.logger.info(f"    multinomial: {self._multiclass}")
        self.logger.info(f"    global: {self._global_explanation}")
        self.logger.info(f"    local: {self._local_explanation}")
        self.logger.info(f"    explanation types: {self._explanation_types}")
        self.logger.info(
            f"    optional e. types: {self._optional_explanation_types}"
        )
        self.logger.info(f"    parameters: {self._parameters}")
        self.logger.info(f"    not standalone: {self._requires_predict_method}")
        self.logger.info(f"    Python deps: {self._modules_needed_by_name}")
        self.logger.info(f"    explainer deps: {self._depends_on}")
        self.logger.info(f"    priority: {self._priority}")

        self.logger.info("explainer instance attributes:")
        self.logger.info(f"    explainer params: {self.explainer_params}")
        self.logger.info(f"    common params: {self.params}")
        self.logger.info(f"    DAI params: {self.dai_params}")
        self.logger.info(f"    explainer deps: {self.explainer_deps}")
        self.logger.info(f"    model with predict method: {self.model}")
        self.logger.info(f"    features used by model: {self.used_features}")
        self.logger.info(f"    target labels: {self.labels}")
        self.logger.info(f"    number of target labels: {self.num_labels}")
        self.logger.info(f"    persistence: {self.persistence}")
        self.logger.info(f"    MLI key: {self.mli_key}")
        self.logger.info(f"    DAI username: {self.dai_username}")
        self.logger.info(f"    model entity: {self.model_entity}")
        self.logger.info(f"    dataset entity: {self.dataset_entity}")
        self.logger.info(
            f"    validation dataset entity: {self.validset_entity}"
        )
        self.logger.info(f"    test dataset entity: {self.testset_entity}")
        self.logger.info(f"    sanitization map: {self.sanitization_map}")
        self.logger.info(f"    enable MOJO: {self.enable_mojo}")
        self.logger.info(f"    Driverless AI configuration: {self.config}")

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]
