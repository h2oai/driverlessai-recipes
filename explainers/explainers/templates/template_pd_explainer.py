# Copyright 2017-2020 H2O.ai, Inc. All rights reserved.
import json
import os
import random

import datatable as dt

from h2oaicore.mli.oss.byor.core.explainers import (
    CustomExplainer,
    CustomExplainerParam,
)
from h2oaicore.mli.oss.byor.core.explanations import (
    CustomExplanation,
    IndividualConditionalExplanation,
    PartialDependenceExplanation,
    WorkDirArchiveExplanation,
)
from h2oaicore.mli.oss.byor.core.representations import (
    IceJsonDatatableFormat,
    PartialDependenceJSonFormat,
    WorkDirArchiveZipFormat,
)
from h2oaicore.mli.oss.byor.explainer_utils import CustomExplainerArgs
from h2oaicore.mli.oss.commons import ExplainerModel, ExplainerParamType


class TemplatePartialDependenceExplainer(CustomExplainer):
    """PD and ICE explainer template.

    Use this template to create explainer with partial dependence (global)
    and individual conditional explanations (local) explanations.

    """

    PARAM_FEATURES = "features"

    _display_name = "Template PD/ICE explainer"
    _description = (
        "PD and ICE explainer template which can be used to create example with "
        "partial dependence (global) and individual conditional explanations "
        "(local) explanations."
    )
    _regression = True
    _binary = True
    _multiclass = True
    _global_explanation = True
    _local_explanation = True
    _explanation_types = [
        PartialDependenceExplanation,
        IndividualConditionalExplanation,
        WorkDirArchiveExplanation,
    ]
    _parameters = [
        # list of features for which should be PD/ICE calculated
        CustomExplainerParam(
            param_name=PARAM_FEATURES,
            description="List of features for which to compute PD/ICE.",
            param_type=ExplainerParamType.multilist,
            src=CustomExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _keywords = [CustomExplainer.KEYWORD_TEMPLATE]

    def __init__(self):
        CustomExplainer.__init__(self)
        self.args = {}

    def setup(self, model: ExplainerModel, persistence, **kwargs):
        CustomExplainer.setup(
            self, model=model, persistence=persistence, **kwargs
        )

        # resolve explainer parameters to instance field
        self.args = CustomExplainerArgs(
            TemplatePartialDependenceExplainer._parameters
        )
        self.args.resolve_params(
            explainer_params=CustomExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            )
        )

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Create global and local (pre-computed/cached) explanations.

        Template explainer returns MOCK explanation data - replace mock data
        preparation with actual computation to create real explainer.

        """
        # explanations list
        explanations = list()

        # global explanation
        pd_explanation = self._explain_pd()
        explanations.append(pd_explanation)

        # local explanation
        ice_explanation = self._explain_ice(X.shape[0])
        pd_explanation.has_local = ice_explanation.explanation_type()
        explanations.append(ice_explanation)

        # work dir archive
        explanations.append(self._explain_zip())

        return explanations

    #
    # global
    #

    def _explain_pd(self):
        """Create global feature importance explanation with JSon (datatable and CSV)
        format representations. JSon representation is supported by Grammar of MLI
        and will be rendered in UI.

        """
        # use parameters
        self.features = (
            TemplatePartialDependenceExplainer.MOCK_FEATURES
            if not self.args.get(self.PARAM_FEATURES)
            else [
                feature
                for feature in TemplatePartialDependenceExplainer.MOCK_FEATURES
                if feature in self.args.get(self.PARAM_FEATURES)
            ]
        )

        # create explanation w/ datatable representation
        global_explanation = PartialDependenceExplanation(
            explainer=self,
            display_name="Template PD/ICE",
            display_category=CustomExplanation.DISPLAY_CAT_EXAMPLE,
        )

        # format: JSon

        # index file
        (
            index_dict,
            index_str,
        ) = PartialDependenceJSonFormat.serialize_index_file(
            features=self.features,
            classes=TemplatePartialDependenceExplainer.MOCK_CLASSES,
            features_meta={"categorical": [self.features[0]]},
            metrics=[{"RMSE": 0.029}, {"SD": 3.1}],
            doc=TemplatePartialDependenceExplainer._description,
        )

        json_representation = PartialDependenceJSonFormat(
            explanation=global_explanation, json_data=index_str
        )
        # data files: per-feature, per-class (saved as added to format)
        # (feature and class names MUST fit names from index file ^)
        for fi, feature in enumerate(self.features):
            for ci, clazz in enumerate(
                TemplatePartialDependenceExplainer.MOCK_CLASSES
            ):
                json_representation.add_data(
                    # IMPROVE: tweak values for every class (1 data for simplicity)
                    format_data=json.dumps(
                        TemplatePartialDependenceExplainer.JSON_FORMAT_DATA
                    ),
                    # filename must fit the name from index file ^
                    file_name=f"pd_feature_{fi}_class_{ci}.json",
                )
        # file with representation can be also copied from work/temp directory
        # json_format.add_file("/tmp/pd-metadata.json", "metadata.json")
        global_explanation.add_format(explanation_format=json_representation)

        return global_explanation

    #
    # local
    #

    def _explain_ice(self, rows: int) -> IndividualConditionalExplanation:
        """Create ICE (local) explanation with JSon+datatable format
        representation. This representation is supported by Grammar of MLI and will
        be rendered in UI.

        As local explanation will be precomputed and persisted (cached), it will be
        returned by Driverless AI automatically. Therefore this explainer doesn't
        have to implement `explain_local()` method for on-demand handling.

        Parameters
        ----------
        rows: int
          Dataset row count for which mock local explanation should be created.

        """
        icejd = IceJsonDatatableFormat
        bins = [
            item["bin"]
            for item in TemplatePartialDependenceExplainer.JSON_FORMAT_DATA[
                "data"
            ]
        ]

        # explanation
        ice_explanation = IndividualConditionalExplanation(
            explainer=self,
            display_name="Template ICE",
            display_category=CustomExplanation.DISPLAY_CAT_EXAMPLE,
        )

        # representation: JSon+datatable
        json_dt_representation = IceJsonDatatableFormat(
            explanation=ice_explanation,
            json_data=IceJsonDatatableFormat.serialize_on_demand_index_file(
                {icejd.KEY_ON_DEMAND: False, icejd.KEY_Y_FILE: icejd.KEY_Y_FILE}
            ),
        )
        ice_explanation.add_format(json_dt_representation)

        # index file
        (index_dict, _) = IceJsonDatatableFormat.serialize_index_file(
            features=self.features,
            classes=TemplatePartialDependenceExplainer.MOCK_CLASSES,
            features_meta=None,
            metrics=None,
            y_file=IceJsonDatatableFormat.FILE_Y_FILE,
        )
        json_dt_representation = IceJsonDatatableFormat(
            explanation=ice_explanation,
            json_data=json.dumps(index_dict, indent=4),
        )

        # data files: per-feature/per-class
        for feature in self.features:
            for cls in TemplatePartialDependenceExplainer.MOCK_CLASSES:
                # generate ICE mock datatable frames
                ice_frame: dt.Frame = (
                    TemplatePartialDependenceExplainer._mock_ice_dt(
                        bins=bins, rows=rows
                    )
                )
                json_dt_representation.add_data_frame(
                    format_data=ice_frame,
                    file_name=index_dict[icejd.KEY_FEATURES][feature][
                        icejd.KEY_FILES
                    ][cls],
                )
        # data files: predictions
        json_dt_representation.add_data_frame(
            format_data=TemplatePartialDependenceExplainer._mock_ice_y_hat(
                rows
            ),
            file_name=IceJsonDatatableFormat.FILE_Y_FILE,
        )

        return ice_explanation

    def _explain_zip(self) -> WorkDirArchiveExplanation:
        explanation = WorkDirArchiveExplanation(
            explainer=self,
            display_name="Template PD/ICE ZIP",
            display_category=CustomExplanation.DISPLAY_CAT_EXAMPLE,
        )
        explanation.add_format(WorkDirArchiveZipFormat(explanation=explanation))
        return explanation

    #
    # datatable PD/ICE
    #

    MOCK_FEATURES = ["feature_1", "feature_2"]
    MOCK_LTYPES = ["str", "real"]
    MOCK_CLASSES = ["class_A", "class_B", "class_C"]

    def _mock_ice_json_dt(
        self,
        explanation: IndividualConditionalExplanation,
        features_meta: dict,
        frame_file: str,
    ):
        # index file
        (index_dict, index_str) = IceJsonDatatableFormat.serialize_index_file(
            features=self.features,
            classes=TemplatePartialDependenceExplainer.MOCK_CLASSES,
            features_meta=features_meta,
            metrics=None,
        )
        json_dt_representation = IceJsonDatatableFormat(
            explanation=explanation, json_data=index_str
        )

        # data files: copy and rename
        icejd = IceJsonDatatableFormat
        for feature in self.features:
            for cls in TemplatePartialDependenceExplainer.MOCK_CLASSES:
                src_file: str = frame_file
                dst_file: str = index_dict[icejd.KEY_FEATURES][feature][
                    icejd.KEY_FILES
                ][cls]
                json_dt_representation.add_file(
                    format_file=src_file, file_name=dst_file
                )
        json_dt_representation.add_file(
            format_file=os.path.join(
                self.persistence.get_explainer_working_dir(),
                IceJsonDatatableFormat.FILE_Y_FILE,
            ),
            file_name=IceJsonDatatableFormat.FILE_Y_FILE,
        )
        return json_dt_representation

    @staticmethod
    def _mock_ice_dt(bins: list, rows: int):
        """Generate mock ICE datatable frame."""

        data_dict: dict = {}
        for bin_ in bins:
            data_dict[str(bin_)] = [random.random() for _ in range(rows)]

        return dt.Frame(data_dict)

    @staticmethod
    def _mock_ice_y_hat(rows: int):
        """Generate mock ICE predictions datatable frame."""
        data: dict = {}
        for clazz in TemplatePartialDependenceExplainer.MOCK_CLASSES:
            data[clazz] = [random.random() for _ in range(rows)]
        return dt.Frame(data)

    #
    # JSon PD/ICE mock
    #

    JSON_FORMAT_DATA: dict = {
        "data": [
            {
                "bin": -2,
                "pd": 0.183_157_429_099_082_95,
                "sd": 0.129_712_074_995_040_9,
                "histogram": 0,
                "oor": True,
            },
            {
                "bin": -1,
                "pd": 0.186_587_452_888_488_77,
                "sd": 0.130_905_747_413_635_25,
                "histogram": 0,
                "oor": True,
            },
            {
                "bin": 0,
                "pd": 0.180_128_112_435_340_88,
                "sd": 0.125_265_195_965_766_9,
                "histogram": 7,
                "oor": False,
            },
            {
                "bin": 1,
                "pd": 0.230_644_226_074_218_75,
                "sd": 0.126_864_895_224_571_23,
                "histogram": 28,
                "oor": False,
            },
            {
                "bin": 2,
                "pd": 0.560_025_811_195_373_5,
                "sd": 0.154_531_568_288_803_1,
                "histogram": 10,
                "oor": False,
            },
            {
                "bin": 3,
                "pd": 0.557_906_091_213_226_3,
                "sd": 0.155_603_677_034_378_05,
                "histogram": 311,
                "oor": False,
            },
            {
                "bin": 4,
                "pd": 0.540_618_479_251_861_6,
                "sd": 0.149_205_371_737_480_16,
                "histogram": 31,
                "oor": False,
            },
            {
                "bin": 5,
                "pd": 0.534_458_696_842_193_6,
                "sd": 0.146_087_795_495_986_94,
                "histogram": 166,
                "oor": False,
            },
            {
                "bin": 6,
                "pd": 0.534_458_696_842_193_6,
                "sd": 0.146_087_795_495_986_94,
                "histogram": 15,
                "oor": False,
            },
            {
                "bin": 7,
                "pd": 0.534_458_696_842_193_6,
                "sd": 0.146_087_795_495_986_94,
                "histogram": 0,
                "oor": True,
            },
            {
                "bin": 8,
                "pd": 0.508_553_981_781_005_9,
                "sd": 0.136_467_516_422_271_73,
                "histogram": 0,
                "oor": True,
            },
        ]
    }
