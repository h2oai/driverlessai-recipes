import numpy as np
import pandas as pd
import os

from h2oaicore.mli.oss.byor.core.explainers import (
    CustomExplainer,
    CustomDaiExplainer,
)
from h2oaicore.mli.oss.byor.core.representations import (
    MarkdownFormat,
    WorkDirArchiveZipFormat,
)
from h2oaicore.mli.oss.commons import ExplainerModel
import datatable as dt

from h2oaicore.mli.oss.byor.core.explanations import (
    AutoReportExplanation,
    WorkDirArchiveExplanation,
)
from h2oaicore.models import try_load_fitted_model
from typing import Optional

from h2oaicore.systemutils import call_subprocess_onetask, config

from h2oaicore import systemutils

from h2oaicore.transformer_utils import (
    preds_columns,
    sanitize_frame,
)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class ActualVsPredictedExplainer(CustomExplainer, CustomDaiExplainer):
    _display_name = "Actual vs Predicted"
    _regression = True
    _binary = False
    _global_explanation = True
    _explanation_types = [AutoReportExplanation]
    _max_zero = True
    _time_series = True
    _img_dir = "images"

    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)

        self.valid = True
        self.target: Optional[str] = None
        self.time_col: Optional[str] = None
        self.tgc: Optional[list] = None
        self.gc: Optional[list] = None
        self.pred_horiz: Optional[int] = None
        self.data_test_periods: Optional[int] = None
        self.images_dir: Optional[str] = None

        self.all_train_dt: Optional[dt.Frame] = None
        self.relevant_train_pd: Optional[pd.DataFrame] = None
        self.train_predictions_pd: Optional[pd.DataFrame] = None

        self.all_test_dt: Optional[dt.Frame] = None
        self.relevant_test_pd: Optional[pd.DataFrame] = None
        self.test_predictions_pd: Optional[pd.DataFrame] = None
        self.test_shapley_pd: Optional[pd.DataFrame] = None

        self.data_df: Optional[pd.DataFrame] = None
        self.data_test_df: Optional[pd.DataFrame] = None
        self.data_test_shapley: Optional[pd.DataFrame] = None

    def easy_case_data(self):
        if self.pred_horiz != self.data_test_periods:
            return None, None, None

        data_df = pd.concat(
            [self.relevant_train_pd, self.train_predictions_pd], axis=1
        )
        data_test_df_shapley = pd.concat(
            [self.relevant_test_pd, self.test_shapley_pd], axis=1
        )
        data_test_df = pd.concat(
            [self.relevant_test_pd, self.test_predictions_pd], axis=1
        )

        return data_df, data_test_df, data_test_df_shapley

    def get_data(self, data_path):
        data_df = dt.fread(data_path)
        subset_data_df = data_df[:, [self.target] + self.tgc].to_pandas()
        subset_data_df[self.time_col] = pd.to_datetime(
            subset_data_df[self.time_col]
        )
        return data_df, subset_data_df

    def get_predictions(self, preds_csv_name):
        pred_path = self.model_entity.fitted_model_path
        pred_path = os.path.join(
            systemutils.data_dir(), os.path.dirname(pred_path), preds_csv_name
        )
        pred_df = pd.read_csv(pred_path)
        pred_df.columns = [
            x.replace(self.target, "prediction") for x in pred_df.columns
        ]

        if self._max_zero:
            pred_df["prediction.lower"] = np.where(
                pred_df["prediction.lower"] < 0, 0, pred_df["prediction.lower"]
            )
        return pred_df

    def get_shapley(self, dt_frame):
        model_path = self.model_entity.fitted_model_path
        fitted_model = try_load_fitted_model(fitted_model_path=model_path)
        if fitted_model is None:
            raise RuntimeError(
                f"Unable to load fitted " f"model from {model_path}"
            )
        if fitted_model.has_pred_contribs:
            print("Model does not support shapley values")
            # raise RuntimeError("Model does not support shapley values")

        X = sanitize_frame(self.logger, dt_frame)
        shapley = fitted_model.predict_safe(
            X, pred_contribs=True, fast_approx=config.mli_fast_approx,
        )

        transformed_column_names = preds_columns(
            target=self.params.target_col,
            labels=fitted_model.labels,
            transformed_columns=fitted_model.transformed_features,
            pred_contribs=True,
        )
        transformed_column_names = [
            x.replace("contrib_", "", 1) for x in transformed_column_names
        ]
        if self.params.target_col in transformed_column_names:
            transformed_column_names.remove(self.params.target_col)

        shapley_df_transformed_feat = dt.Frame(
            shapley, names=transformed_column_names
        ).to_pandas()
        return shapley_df_transformed_feat

    def setup(self, model: ExplainerModel, persistence, **kwargs):
        CustomExplainer.setup(
            self, model=model, persistence=persistence, **kwargs
        )

        CustomDaiExplainer.setup(self, **kwargs)

        self.valid = True
        if self.testset_entity is None:
            self.valid = False
            return None

        self.target = self.model_entity.parameters.target_col
        self.time_col = self.model_entity.parameters.time_col
        self.tgc = self.model_entity.parameters.time_groups_columns
        self.gc = [x for x in self.tgc if x != self.time_col]
        self.pred_horiz = self.model_entity.parameters.num_prediction_periods

        self.all_train_dt, self.relevant_train_pd = self.get_data(
            self.dataset_entity.bin_file_path
        )

        self.all_test_dt, self.relevant_test_pd = self.get_data(
            self.testset_entity.bin_file_path
        )
        self.data_test_periods = (
            self.relevant_test_pd[self.time_col].unique().shape[0]
        )

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        """Create global and local (pre-computed/cached) explanations.
        Template explainer returns MOCK explanation data - replace mock data
        preparation with actual computation to create real explainer.
        """
        # explanations list

        if not self.valid:
            return []

        self.train_predictions_pd = self.get_predictions("train_preds.csv")
        self.test_predictions_pd = self.get_predictions("test_preds.csv")
        self.test_shapley_pd = self.get_shapley(self.all_test_dt)

        data_df, data_test_df, data_test_df_shapley = self.easy_case_data()

        self.data_df = data_df
        self.data_test_df = data_test_df
        self.data_test_shapley = data_test_df_shapley

        self.images_dir = self.persistence.get_explainer_working_file(
            self._img_dir
        )
        if not os.path.exists(self.images_dir):
            os.mkdir(self.images_dir)

        # global explanation
        return self.explain_global_markdown()

    def explain_global_markdown(self):
        global_explanation = AutoReportExplanation(
            explainer=self,
            display_name="Documentation",
            display_category=AutoReportExplanation.DISPLAY_CAT_AUTOREPORT,
        )

        # CALCULATION: Markdown report with image(s) in work directory
        report_path, images_path = self.plot_actuals_preds(
            self.data_df, self.data_test_df
        )

        global_explanation.add_format(
            MarkdownFormat(
                explanation=global_explanation,
                format_file=report_path,
                extra_format_files=images_path,
            )
        )

        # representation: work dir
        work_explanation = WorkDirArchiveExplanation(
            explainer=self,
            display_name="ZIP",
            display_category=AutoReportExplanation.DISPLAY_CAT_AUTOREPORT,
        )
        work_explanation.add_format(
            WorkDirArchiveZipFormat(explanation=work_explanation)
        )

        explanations = [global_explanation, work_explanation]
        return explanations

    @staticmethod
    def get_cond_val(x):
        if type(x) is str or not np.isfinite(x):
            return f"'{x}'"
        return x

    def plot_actuals_preds(self, df_train, df_test, non_zero=True):
        group_cols = self.gc
        t_col = self.time_col
        tgt = self.target

        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_squared_error as rmse

        all_images = []
        groups = df_train[group_cols].drop_duplicates()

        min_date_val = df_train[~df_train["prediction"].isna()][t_col].min()
        max_date_val = df_train[~df_train["prediction"].isna()][t_col].max()
        min_date_test = df_test[~df_test["prediction"].isna()][t_col].min()
        max_date_test = df_test[~df_test["prediction"].isna()][t_col].max()

        descr = "# Actuals vs Predictions Report\n"

        n_groups = groups.shape[0]
        for i, values in enumerate(groups.values):
            condition = " and ".join(
                [
                    f"{name}=={self.get_cond_val(value)}"
                    for name, value in zip(groups.columns, values)
                ]
            )
            self.logger.info(
                f"Rendering image for tgc {condition} ({i+1}/{n_groups})"
            )
            title = (
                condition.replace(" and", ",")
                .replace("'", "")
                .replace("==", ":")
            )
            # print(condition,title)
            df_train_s = df_train.query(condition).sort_values(t_col).copy()
            df_test_s = df_test.query(condition).sort_values(t_col).copy()

            if non_zero:
                df_train_s["prediction.lower"] = np.where(
                    df_train_s["prediction.lower"] < 0,
                    0,
                    df_train_s["prediction.lower"],
                )
                df_test_s["prediction.lower"] = np.where(
                    df_test_s["prediction.lower"] < 0,
                    0,
                    df_test_s["prediction.lower"],
                )

            plt.figure(figsize=(20, 10))
            plt.plot(df_train_s[t_col], df_train_s[tgt], "k")
            plt.plot(df_train_s[t_col], df_train_s["prediction"], "b--")
            # plt.plot(data_df_s[time_col], data_df_s['prediction_ovf'], 'r-.')
            plt.fill_between(
                df_train_s[t_col],
                df_train_s["prediction.lower"],
                df_train_s["prediction.upper"],
                color="b",
                alpha=0.2,
            )

            plt.plot(df_test_s[t_col], df_test_s[tgt], "k")
            plt.plot(df_test_s[t_col], df_test_s["prediction"], "r--")
            # plt.plot(data_df_s[time_col], data_df_s['prediction_ovf'], 'r-.')
            plt.fill_between(
                df_test_s[t_col],
                df_test_s["prediction.lower"],
                df_test_s["prediction.upper"],
                color="b",
                alpha=0.2,
            )

            plt.axvspan(min_date_val, max_date_val, facecolor="r", alpha=0.1)
            plt.axvspan(min_date_test, max_date_test, facecolor="g", alpha=0.1)
            # plt.axvspan(COVID_start, COVID_end, facecolor='g', alpha=0.1)

            plt.title(title)
            plt.legend([tgt, "prediction_valitdation", tgt, "prediction_test"])
            image_file = f"{title}.svg"
            replace_map = {
                " ": "_",
                ":": "-",
                ",": "",
            }

            for k, v in replace_map.items():
                image_file = image_file.replace(k, v)

            image_path = os.path.join(self.images_dir, image_file)
            all_images.append(image_path)
            plt.savefig(image_path)

            try:
                r2 = round(
                    r2_score(df_test_s[tgt], df_test_s["prediction"]) * 2
                )
                mape_score = round(
                    mape(df_test_s[tgt], df_test_s["prediction"])
                )
            except Exception as e:
                err = f"Failed to compute metrics for {condition} with{str(e)}"
                self.logger.error(err)
                r2 = "NA"
                mape_score = "NA"

            image_file = f"{self._img_dir}/{image_file}"
            self.logger.info(f"Rendered {image_file}")
            descr = (
                f"{descr}"
                f"## {title}"
                f"\nMetric | Score"
                f"\n:---:|:---:"
                f"\nR2 | {r2}%"
                f"\nMAPE | {mape_score}%"
                f"\n\n![]({image_file})\n\n"
            )

        report_path = self.persistence.get_explainer_working_file("report.md")
        with open(report_path, "w") as md_file:
            md_file.write(descr)

        return report_path, all_images
