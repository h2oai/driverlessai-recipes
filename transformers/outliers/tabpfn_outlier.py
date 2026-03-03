"""
TabPFN-based outlier scoring transformer for Driverless AI.

License compliance note (Prior Labs License v1.2, See license text; ensure compliance with attribution requirements):
- This recipe depends on `tabpfn` / `tabpfn-extensions` and may download/use TabPFN weights.
- If you DISTRIBUTE or make available a product/service containing TabPFN source/weights (or derivative work),
    you must satisfy the license additional attribution requirement (Section 10), including prominently displaying:
    “Built with PriorLabs-TabPFN” in relevant UI/docs.
"""
import logging
import os
import pathlib
import time
import random
import uuid
from typing import List, Optional, Sequence, Tuple

import datatable as dt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from h2oaicore import systemutils
from h2oaicore.systemutils_more import download
from h2oaicore.transformer_utils import CustomTransformer


TABPFN_CLASSIFIER_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
)
TABPFN_REGRESSOR_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-regressor.ckpt"
)
TABPFN_CLASSIFIER_CKPT_SHA256 = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
TABPFN_REGRESSOR_CKPT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
MAX_CLASSES = 10


class TabPFNOutliersDetection:
    """
    Inspired from https://github.com/PriorLabs/tabpfn-extensions/blob/a54bc14398a5155ae22a5c0ac2fb9327e88782a8/src/tabpfn_extensions
    Outlier-only subset of TabPFNUnsupervisedModel.
    """

    @staticmethod
    def load_from_bytes(model_bytes) -> "TabPFNOutliersDetection":
        return systemutils.load_obj_bytes(model_bytes)

    def __init__(self, classifier, regressor, num_features: int, eps: float = 1e-10, seed: int = 0, top_features: int = 5, logger=None):
        self._classifier = classifier
        self._regressor = regressor
        self._eps = eps
        self._seed = seed
        self._logger = logger
        self._num_features = num_features

        self.train_x_ = None
        self.ord_encoder_ = None
        self.numeric_col_mask_: Optional[np.ndarray] = None
        # (#features, #K): (i, j) represents top K important feature indices when ith feature as target, heuristically find the top K conditioning set.
        # avoid growing prefix of columns, approximate with fixed size better for caching.
        self.top_features_matrix_ = np.full((self._num_features, min(top_features, num_features - 1)), fill_value=-1, dtype=int)


    def to_bytes(self):
        return systemutils.save_obj_to_bytes(self)

    def fit(self, X: dt.Frame, y=None):
        assert len(X.shape) == 2

        self.ord_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        self.numeric_col_mask_ = _numeric_column_mask(X)
        x_numpy = _to_numeric(X, self.ord_encoder_, numeric_col_mask=self.numeric_col_mask_)
        self.train_x_ = np.asarray(x_numpy, dtype=np.float32)
        del x_numpy

        # Top-k relevant features used in scoring.
        self._fit_with_full_surrogate_rf(self.train_x_)

        return self

    def outliers(
        self,
        x: dt.Frame,
        n_permutations: int = 10,
        fast_mode: bool = False,
        seed: int = 0,
    ) -> np.ndarray:
        """Calculate the negative logarithm outlier scores for each sample in the input data.

        This is the implementation for outlier detection, which calculates
        sample probability for each sample in X by multiplying the probabilities of
        each feature according to chain rule of probability. And take the negative
        logarithm of the final resulting probability. higher scores (lower probabilities)
        indicate more likely outliers.

        Parameters:
            x: dt.Frame
                Samples to calculate outlier scores for, shape (n_samples, n_features)
            n_permutations: int, default=10
                Number of permutations to use for more robust probability estimates.
                Higher values may produce more stable results but increase computation time.
            fast_mode: bool, default=False
                Whether in test mode to do the feature permutations.
            seed: int, default=0
                Random seed

        Returns:
            np.ndarray:
                Array of negative logarithm outlier scores (higher values indicate more likely outliers),
                shape (n_samples,)

        Raises:
            RuntimeError: If the model initialization fails
            ValueError: If the input data has incompatible dimensions
        """
        assert len(x.shape) == 2
        check_is_fitted(self, attributes=["train_x_", "ord_encoder_", "top_features_matrix_"])

        seed = self._seed if seed is None else seed
        x_numpy = _to_numeric(x, self.ord_encoder_, numeric_col_mask=self.numeric_col_mask_)
        x_numpy = np.asarray(x_numpy, dtype=np.float32)

        n_features = x.shape[1]
        all_features = list(range(n_features))

        actual_n_permutations = 1 if fast_mode else n_permutations
        log_densities = []

        # Run outlier scoring per permutation
        st = time.perf_counter()
        permutations = _efficient_random_permutation(all_features, actual_n_permutations, seed)
        n_perms = len(permutations)
        for perm_idx, perm in enumerate(permutations):
            perm_st = time.perf_counter()
            systemutils.loggerinfo(self._logger,
                                   f"[{self.__class__.__name__}] [outliers] Starting permutation {perm_idx + 1}/{n_perms}")
            perm_density_log = self.outliers_single_permutation_(
                x=x_numpy,
                feature_permutation=perm,
                seed=seed,
            )

            perm_density_log = np.nan_to_num(perm_density_log, nan=-1e30, posinf=0.0, neginf=-1e30)
            log_densities.append(perm_density_log)

            del perm_density_log
            systemutils.loggerinfo(self._logger,
                                   f"[{self.__class__.__name__}] [outliers] Permutation {perm_idx + 1}/{n_perms} "
                                   f"took {(time.perf_counter() - perm_st):.1f}s")
        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [outliers] Outlier scoring takes {(time.perf_counter() - st):.6f} seconds")

        self._claim_memory()

        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [outliers] Permutate features finished!")

        log_densities_array = np.stack(log_densities, axis=0)
        del log_densities

        avg_log_density = _log_mean_exp(log_densities_array, axis=0)
        final_scores = -avg_log_density

        self._claim_memory(force=True)

        return final_scores

    def outliers_single_permutation_(
        self,
        x: np.ndarray,
        feature_permutation: Tuple[int, ...],
        seed: Optional[int] = None,
    ):
        import torch

        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [outliers_single_permutation_] Outlier score start")
        perm_wall_start = time.perf_counter()
        # Start with a log probability of 0 (log(1) = 0)
        log_p = np.zeros(x[:, 0].shape, dtype=np.float64)
        seed = self._seed if seed is None else seed

        for i, column_idx in enumerate(feature_permutation):
            model, x_predict, y_predict, is_classification = self.density_(
                x_predict=x,
                x_fit=self.train_x_,
                conditional_idx=self.top_features_matrix_[column_idx],
                column_idx=column_idx,
                seed=seed + 1000 * i + int(column_idx),
            )
            if is_classification:
                st = time.perf_counter()
                pred_np = model.predict_proba(x_predict)
                systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [outliers_single_permutation_] TabPFN probability prediction took {(time.perf_counter() - st):.6f} seconds")

                # Map y_predict values to predict_proba column indices via model.classes_.
                # Direct integer indexing (y_predict.astype(int) as column index) is WRONG
                # when class labels are not 0-based contiguous (e.g., numeric columns with
                # values like [100, 200, 300], or ordinal-encoded columns with missing=-2).
                classes = model.classes_
                y_vals = y_predict.astype(classes.dtype)
                col_indices = np.searchsorted(classes, y_vals)
                # searchsorted can return out-of-bounds; validate exact match
                col_indices_clipped = np.minimum(col_indices, len(classes) - 1)
                valid = (col_indices < len(classes)) & (classes[col_indices_clipped] == y_vals)

                pred = np.full(x_predict.shape[0], self._eps, dtype=np.float32)
                valid_rows = np.where(valid)[0]
                pred[valid_rows] = pred_np[valid_rows, col_indices[valid_rows]]

                # Clip to [eps, 1.0] to handle both underflow and potential numerical issues
                pred = np.clip(pred, self._eps, 1.0)

                del pred_np, classes, y_vals, col_indices, col_indices_clipped, valid, valid_rows
            else:
                st = time.perf_counter()
                # Regression: use proper Gaussian likelihood from TabPFN's PDF
                pred = model.predict(x_predict, output_type="full")
                systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [outliers_single_permutation_] TabPFN prediction took {(time.perf_counter() - st):.6f} seconds")

                # Get logits tensor properly
                logits = pred["logits"]
                if hasattr(logits, "detach"):
                    logits_tensor = logits.detach()
                else:
                    logits_tensor = torch.as_tensor(logits)
                y_tensor = torch.tensor(y_predict).to(logits_tensor.device)

                # PDF returns probability density (can be > 1 for regression)
                pred_t = pred["criterion"].pdf(logits_tensor, y_tensor)
                pred = pred_t.detach().cpu().numpy().astype(np.float32)

                # For regression, densities can theoretically be > 1
                # Clip to prevent log of zero or negative values
                pred = np.clip(pred, self._eps, None)

                del logits, logits_tensor, y_tensor, pred_t

            log_pred = np.log(pred)

            log_p = log_p + log_pred

            del model, x_predict, y_predict, pred, log_pred

            if i > 0 and i % 5 == 0:
                self._claim_memory(force=True)

        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [outliers_single_permutation_] Outlier scores permutations takes {(time.perf_counter() - perm_wall_start):.6f} seconds")
        return log_p

    def density_(
        self,
        x_predict: np.ndarray,
        x_fit: np.ndarray,
        conditional_idx: Tuple[int, ...],
        column_idx: int,
        seed: Optional[int] = None,
    ):
        """Generate density predictions for a specific feature based on other features.

        This internal method is used by the imputation and outlier detection algorithms
        to model the conditional probability distribution of one feature given others.

        Args:
            x_predict: Input data for which to make predictions
            x_fit: Training data to fit the model
            conditional_idx: Indices of features to condition on
            column_idx: Index of the feature to predict
            seed: Random seed, default=0

        Returns:
            tuple containing:
                - The fitted model (classifier or regressor)
                - The filtered features used for prediction
                - The target feature values to predict
                - Whether model is classification
        """
        y_fit = x_fit[:, column_idx]
        is_classification = self._is_classification(y_fit, MAX_CLASSES)

        st = time.perf_counter()
        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [density] model fitting preparations started")
        # conditional_idx always has ≥1 entry because top_features_matrix_ has
        # shape (num_features, min(top_features, num_features-1)) and min_cols=2
        # guarantees num_features ≥ 2, so every feature has at least 1 conditioning feature.
        x_fit = x_fit[:, conditional_idx]
        x_fit = x_fit.reshape(x_fit.shape[0], -1)

        x_predict, y_predict = x_predict[:, conditional_idx], x_predict[:, column_idx]
        x_predict = x_predict.reshape(x_predict.shape[0], -1)

        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [density] model fitting preparations ended takes {(time.perf_counter() - st):.6f} seconds")
        # Handle potential nan values in y_fit
        if np.isnan(y_fit).any():
            y_fit = np.nan_to_num(y_fit, nan=0.0)

        st = time.perf_counter()
        # clone() creates a fresh unfitted estimator, invalidating fit_with_cache.
        # Each clone+fit is a cold-start TabPFN fit. This is architecturally
        # required by the SAGE permutation algorithm (each feature subset
        # needs an independent model), but is the main performance bottleneck.
        model = clone(self._get_model(is_classification))
        systemutils.loggerinfo(self._logger,
                               f"[{self.__class__.__name__}] [density] model clone takes {(time.perf_counter() - st):.6f} seconds")
        if is_classification:
            y_fit = y_fit.astype(np.int64)
            y_predict = y_predict.astype(np.int64)

        st = time.perf_counter()
        systemutils.loggerinfo(self._logger,
                               f"[{self.__class__.__name__}] [density] model fitting started")
        model.fit(x_fit, y_fit)
        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [density_] TabPFN fitting takes {(time.perf_counter() - st):.6f} seconds")

        return model, x_predict, y_predict, is_classification

    def _fit_with_full_surrogate_rf(self, _x_train: np.ndarray):
        """Surrogate RF to construct subset features"""
        p = self._num_features
        k = self.top_features_matrix_.shape[1]
        all_cols = np.arange(p, dtype=int)

        st = time.perf_counter()
        for column_idx in range(p):
            pred_cols = np.delete(all_cols, column_idx)
            targets = _x_train[:, column_idx]
            features = _x_train[:, pred_cols]
            top_mask = self._get_top_k_single_surrogate_rf(targets=targets, features=features, k=k)
            self.top_features_matrix_[column_idx, :] = pred_cols[top_mask]
        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [_fit_with_full_surrogate_rf] fit the top_features_matrix takes {(time.perf_counter() - st):.6f} seconds")

    def _get_top_k_single_surrogate_rf(self, targets: np.ndarray, features: np.ndarray, k: int) -> np.ndarray:
        """Top k relevant features"""
        if np.isnan(targets).any():
            targets = np.nan_to_num(targets, nan=0.0)

        if self._is_classification(targets, MAX_CLASSES):
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features="sqrt", min_samples_leaf=2, random_state=self._seed, n_jobs=-1)
        else:
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, max_features="sqrt", min_samples_leaf=2, random_state=self._seed, n_jobs=-1)

        st = time.perf_counter()
        rf.fit(features, targets)
        systemutils.loggerinfo(self._logger, f"[{self.__class__.__name__}] [_get_top_k_single_surrogate_rf] fitting one target column from top features matrix takes {(time.perf_counter() - st):.6f} seconds")

        importance = np.asarray(rf.feature_importances_, dtype=np.float64)
        k_eff = min(k, len(importance))
        top = np.argpartition(-importance, kth=k_eff - 1)[:k_eff]
        top = top[np.argsort(-importance[top])]
        return top.astype(int, copy=False)

    def _get_model(self, is_classification: bool):
        return self._classifier if is_classification else self._regressor

    @staticmethod
    def _is_classification(targets: np.ndarray, upper_bound: int = -1) -> bool:
        try:
            check_classification_targets(targets)
        except ValueError:
            return False

        if upper_bound < 0:
            return 1 < np.unique(targets).size
        else:
            return 1 < np.unique(targets).size <= MAX_CLASSES

    @staticmethod
    def _claim_memory(force: bool = False):
        import gc
        import torch

        use_gpu = torch.cuda.is_available()
        if force:
            if use_gpu:
                torch.cuda.empty_cache()
        else:
            if use_gpu:
                try:
                    max_allocated = torch.cuda.max_memory_allocated()
                    if not max_allocated:
                        # Avoid division by zero; fall back to conservative cleanup
                        torch.cuda.empty_cache()
                        gc.collect()
                        return
                    allocated_memory = torch.cuda.memory_allocated() / max_allocated
                    if allocated_memory > 0.8:
                        torch.cuda.empty_cache()
                except (RuntimeError, AttributeError):
                    # Fallback if memory stats unavailable
                    torch.cuda.empty_cache()
        gc.collect()


class TabPFNOutlierScoreTransformer(CustomTransformer):
    r"""
        TabPFN-based outlier score transformer for Driverless AI.

        CAUTION: TabPFN pretrained model has fitting size limitation, the upperbound to be max_fit_rows <= 10000,
        > 10000 is technically feasible but performance can be jeopardized.
        GPU inference is highly recommend and this transformer works best to small dataset < 10000
        reference: https://github.com/PriorLabs/TabPFN

        Due to the running complexity of computing posterior using chain of probabilities, this transformer is highly recommended
        to only used for unsupervised learning.

        **What it does**
        - Fits an unsupervised TabPFN-based density estimator on the selected feature columns.
        - Produces an **outlier score per row** based on the estimated likelihood under the model.
          The underlying detector returns an averaged density estimate across random feature permutations.
        - This transformer converts that density to a more intuitive anomaly score:
          \[
          \text{score}(x) = -\log(\max(p(x), \epsilon))
          \]
          so that **higher scores indicate more anomalous rows**.

        **How it works (high level)**
        - For each permutation of features, estimate per-feature conditional probabilities and multiply them
          via the chain rule to get a joint density proxy.
        - Average densities across permutations for robustness (configurable via `n_permutations`).

        **References**
        - Upstream TabPFN unsupervised utilities (outliers / imputation / synthetic):
          https://github.com/PriorLabs/tabpfn-extensions/blob/a54bc14398a5155ae22a5c0ac2fb9327e88782a8/src/tabpfn_extensions/unsupervised/unsupervised.py
        - TabPFN project:
          https://github.com/PriorLabs/TabPFN

        **Weights**
        - Classifier checkpoint:
          {TABPFN_CLASSIFIER_CKPT_URL}
        - Regressor checkpoint:
          {TABPFN_REGRESSOR_CKPT_URL}

        **License / attribution (Prior Labs License v1.2, See license text; ensure compliance with attribution requirements)**
        This transformer relies on TabPFN and may download/use TabPFN model weights.
        The TabPFN license is Apache 2.0 **with an additional attribution provision**.
        If you **distribute or make available** a product/service that contains TabPFN source or model weights,
        or a derivative work relating to them, you must:
        1) provide a copy of the license, and
        2) prominently display: **"Built with PriorLabs-TabPFN"** on relevant UI/docs pages.
        (Internal benchmarking/testing without external communication does not qualify as distribution per the license.)

        Notes for DAI users:
        - This transformer is computationally expensive; keep `max_cols` and `n_permutations` conservative.
        - Output is numeric and can be used directly as an engineered feature for downstream models.
        """
    _unsupervised = True
    _regression = True
    _binary = True
    _multiclass = True
    _numeric_output = True
    _is_reproducible = True
    _parallel_task = False
    _can_use_gpu = True
    _must_use_gpu = True
    _can_use_multi_gpu = False
    _get_gpu_lock = True
    _get_gpu_lock_vis = True
    _mojo = False
    _testing_can_skip_failure = False
    _allow_transform_to_modify_output_feature_names = False
    _modules_needed_by_name = ["tabpfn==6.4.1"]

    TRAIN_SIZE_LIMITS = 10000
    TRAIN_SIZE_OVERLOAD_RATE = 2
    MAX_FEATURES = 15

    @staticmethod
    def do_acceptance_test():
        # Very slow, manually set be `False` to skip
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=2, max_cols=TabPFNOutlierScoreTransformer.MAX_FEATURES, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return dict(
            n_permutations=[5],
            n_estimators=[10],
            softmax_temperature=[0.5],
            balance_probabilities=[True],
            average_before_softmax=[True],
            eps=[1e-10],
        )

    def __init__(
        self,
        num_cols: Sequence[str] = (),
        output_features_to_drop: Sequence[str] = (),
        n_permutations: int = 5,
        n_estimators: int = 8,
        softmax_temperature: float = 0.5,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        eps: float = 1e-12,
        top_features: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        init_args_dict = locals().copy()
        self.params = {k: v for k, v in init_args_dict.items() if k in self.get_parameter_choices()}
        self._output_features_to_drop = output_features_to_drop

        self.max_fit_rows = self.TRAIN_SIZE_LIMITS
        self.uid = str(uuid.uuid4())
        self.seed = systemutils.config.seed
        self.top_features = top_features

    def fit_transform(self, X: dt.Frame, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        assert len(X.shape) == 2

        logger = None
        if self.context and self.context.experiment_id:
            logger = systemutils.make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        self._prepare_env(seed=self.seed)
        x, sample_indices = self._prepare_x(x=X, density_sampling=True, logger=logger)

        tabpfn_classifier, tabpfn_regressor = self._build_tabpfn_models(
            seed=self.seed,
            n_jobs=self._get_n_jobs(logger, **kwargs),
            n_estimators=self.params['n_estimators'],
            device=self._get_device(),
            softmax_temperature=self.params['softmax_temperature'],
            balance_probabilities=self.params['balance_probabilities'],
            average_before_softmax=self.params['average_before_softmax'],
        )
        self.detector_ = TabPFNOutliersDetection(
            classifier=tabpfn_classifier,
            regressor=tabpfn_regressor,
            num_features=x.shape[1],
            eps=self.params['eps'],
            top_features=self.top_features,
            seed=self.seed,
            logger=logger,
        )
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Fitting outlier detector...")
        st = time.perf_counter()
        self.detector_.fit(x[sample_indices, :])
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Finished fitting outlier detector, takes {(time.perf_counter() - st):.6f} seconds")

        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Scoring outlier detector...")
        st = time.perf_counter()
        raw_scores = self._scores(x)
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Finished scoring outlier detector, takes {(time.perf_counter() - st):.6f} seconds")

        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Calibrating scores...")
        st = time.perf_counter()
        # Calibrate scores on training data for interpretability
        calibrated_scores = self._calibrate_scores(raw_scores, training=True)
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Finished calibrating scores, takes {(time.perf_counter() - st):.6f} seconds")

        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Transforming score output...")
        st = time.perf_counter()
        final_output = self._transform(scores=calibrated_scores)
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Finished transforming score output, takes {(time.perf_counter() - st):.6f} seconds")

        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Saving model...")
        st = time.perf_counter()
        self._save_model()
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit_transform] Finished saving model, takes {(time.perf_counter() - st):.6f} seconds")
        return final_output

    def transform(self, X: dt.Frame, **kwargs) -> np.ndarray:
        assert len(X.shape) == 2
        check_is_fitted(self, ["raw_model_bytes"])
        self.detector_ = TabPFNOutliersDetection.load_from_bytes(self.raw_model_bytes)
        assert self.detector_ is not None

        logger = None
        if self.context and self.context.experiment_id:
            logger = systemutils.make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        self._prepare_env(seed=self.seed)
        x, _ = self._prepare_x(X)

        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [transform] Scoring model with {self.params['n_permutations']} permutations...")
        st = time.perf_counter()
        raw_scores = self._scores(x)
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [transform] Finished scoring model with {self.params['n_permutations']} permutations, takes {(time.perf_counter() - st):.6f} seconds")

        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [transform] Calibrating scores...")
        st = time.perf_counter()
        # Apply calibration using parameters from training
        calibrated_scores = self._calibrate_scores(raw_scores, training=False)
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [transform] Finished calibrating scores, takes {(time.perf_counter() - st):.6f} seconds")

        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [transform] Transforming model...")
        st = time.perf_counter()
        final_output = self._transform(scores=calibrated_scores)
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [transform] Finished transforming model, takes {(time.perf_counter() - st):.6f} seconds")
        return final_output

    def _transform(self, scores: np.ndarray) -> np.ndarray:
        # _scores() always scores the full frame, so no padding needed.
        return scores.reshape(-1)

    def _scores(self, x: dt.Frame) -> np.ndarray:
        scores = self.detector_.outliers(
            x=x,
            n_permutations=self.params['n_permutations'],
            seed=self.seed,
        )
        assert scores.shape[0] == x.shape[0]
        return scores

    def _save_model(self):
        self.raw_model_bytes = self.detector_.to_bytes()

    def _calibrate_scores(self, raw_scores: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Calibrate outlier scores to [0, 1] range with probabilistic interpretation.

        Uses training data percentiles for stable calibration. Scores become
        interpretable as "probability of being an outlier".

        Args:
            raw_scores: Raw negative log likelihood scores from outlier detection
            training: If True, fit calibration parameters; if False, apply existing parameters

        Returns:
            Calibrated scores in [0, 1] range where higher values indicate higher outlier probability
        """
        if training:
            # Store calibration parameters from training data
            self.score_percentiles_ = np.percentile(raw_scores, [1, 5, 25, 50, 75, 95, 99])
            self.score_median_ = self.score_percentiles_[3]  # 50th percentile
            # Use IQR-like robust scale: difference between 95th and 5th percentiles
            self.score_scale_ = self.score_percentiles_[5] - self.score_percentiles_[1]
            if self.score_scale_ < 1e-6:
                self.score_scale_ = 1.0  # Avoid division by zero

        # Robust standardization (resistant to outliers)
        calibrated = (raw_scores - self.score_median_) / (self.score_scale_ + 1e-6)

        # Sigmoid to map to [0, 1] with probabilistic interpretation
        return 1.0 / (1.0 + np.exp(-calibrated))

    def _prepare_x(self, x: dt.Frame, density_sampling: bool = False, logger: Optional[logging.Logger] = None) -> Tuple[dt.Frame, np.ndarray]:
        """
        Intelligent sampling that preserves data distribution.

        For unsupervised outlier detection, we use density-aware sampling to ensure
        we capture tail behavior and rare patterns that are crucial for outlier detection.
        """
        n = x.shape[0]
        sample_indices = np.arange(n)

        if n <= self.max_fit_rows:
            return x, sample_indices

        if density_sampling:
            # For outlier detection: density-aware sampling
            # Sample more from sparse regions to capture outliers and edge cases
            sample_indices = self._density_aware_sampling(
                x=x,
                n_samples=self.max_fit_rows,
                rng=np.random.RandomState(self.seed),
                logger=logger,
            )

        return x, sample_indices

    def _density_aware_sampling(
        self,
        x: dt.Frame,
        n_samples: int,
        rng: np.random.RandomState,
        eps: float = 1e-10,
        logger: Optional[logging.Logger] = None,
    ) -> np.ndarray:
        """
        Sample more from low-density regions to capture outliers and edge cases.

        Uses k-NN density estimation for importance weights.
        Low density points (large k-NN distance) get higher sampling probability.

        Args:
            x: Input data frame
            n_samples: Number of samples to select
            rng: Random number generator
            eps: Epsilon, avoid divide by zero
            logger: Logger instance

        Returns:
            Array of selected sample indices
        """
        if x.shape[0] <= n_samples:
            return np.arange(x.shape[0])

        from sklearn.neighbors import NearestNeighbors

        # Convert to numeric for distance calculation
        probe_indices = np.arange(x.shape[0])
        x_probe_numeric = _to_numeric_fast(x.to_numpy())

        # k-NN density estimation (smaller distance = higher density)
        num_classes = len(self.labels or [])
        # num_classes > 1 then classes would be bounded by MAX_CLASSES, otherwise regression takes 1% of the overall dataset as number of target regions
        k = min(MAX_CLASSES, num_classes) if num_classes > 1 else min(100, x.shape[0] // 100)
        k = max(1, k)
        try:
            st = time.perf_counter()
            nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_probe_numeric)
            distances, _ = nn.kneighbors(x_probe_numeric)
            systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [_density_aware_sampling] KNN fitting takes {(time.perf_counter() - st):.6f} seconds")

            # Inverse density as importance weight
            # Low density points (large distance) get higher probability
            density = 1.0 / (distances.mean(axis=1) + eps)
            weights = 1.0 / (density + eps)
            weights /= weights.sum() # Normalize

            # Sample with replacement weighted by inverse density
            selected = rng.choice(probe_indices, size=n_samples, replace=False, p=weights)
            return selected

        except Exception as e:
            systemutils.loggerwarning(None, f"Density-aware sampling failed: {e}. Using uniform sampling.")
            return rng.choice(x.shape[0], n_samples, replace=False)

    @staticmethod
    def _prepare_env(seed: int):
        import torch

        # Only set torch seeds (needed for CUDA determinism in TabPFN).
        # Avoid setting global np.random.seed() / random.seed() to prevent
        # polluting state for other models running in the same process.
        torch.manual_seed(seed)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            torch.cuda.manual_seed_all(seed)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256"

        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(_get_cache_dir())
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    @staticmethod
    def _get_n_jobs(logger, **kwargs) -> int:
        max_workers = kwargs.get('max_workers', None)
        if max_workers is None:
            systemutils.loggerwarning(logger, "No Max Worker in kwargs. Set n_jobs to 1")
            return 1
        try:
            if systemutils.config.fixed_num_folds <= 0:
                n_jobs = max(1, int(
                    systemutils.max_threads() / min(systemutils.config.num_folds, max_workers)))
            else:
                n_jobs = max(1, int(
                    systemutils.max_threads() / min(systemutils.config.fixed_num_folds,
                                                    systemutils.config.num_folds, max_workers)))
        except (KeyError, AttributeError):
            n_jobs = 1

        return n_jobs

    @staticmethod
    def _build_tabpfn_models(
        seed: int,
        n_jobs: int,
        n_estimators: int,
        device: str,
        softmax_temperature: float = 0.5,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
    ):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        clf_ckpt, reg_ckpt = TabPFNOutlierScoreTransformer._ensure_weights_cached()

        systemutils.loggerinfo(None, f"Instantiating TabPFN Classifier and Regressor with device {device}")
        tabpfn_clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device=device,
            model_path=clf_ckpt,
            random_state=seed,
            n_preprocessing_jobs=n_jobs,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            fit_mode="fit_with_cache",  # Faster inference for GPU.
        )
        tabpfn_reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=device,
            model_path=reg_ckpt,
            random_state=seed,
            n_preprocessing_jobs=n_jobs,
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            fit_mode="fit_with_cache",  # Faster inference for GPU.
        )

        return tabpfn_clf, tabpfn_reg

    @staticmethod
    def _ensure_weights_cached():
        """Download checkpoints into a deterministic cache location.

        Uses atomic PID-stamped temp files to prevent concurrent workers from
        reading partially-written files.  Verification happens only on freshly
        downloaded checkpoints (not on every call).
        """
        cache_dir = _get_cache_dir()
        systemutils.makedirs(cache_dir, exist_ok=True)

        clf_path = cache_dir / os.path.basename(TABPFN_CLASSIFIER_CKPT_URL)
        reg_path = cache_dir / os.path.basename(TABPFN_REGRESSOR_CKPT_URL)

        for url, final_path, expected_sha in [
            (TABPFN_CLASSIFIER_CKPT_URL, clf_path, TABPFN_CLASSIFIER_CKPT_SHA256),
            (TABPFN_REGRESSOR_CKPT_URL, reg_path, TABPFN_REGRESSOR_CKPT_SHA256),
        ]:
            if not final_path.exists():
                # Atomic download: write to PID-stamped temp file, verify, then rename.
                tmp_path = final_path.with_suffix(f".tmp.{os.getpid()}")
                try:
                    download(url, dest=str(tmp_path))
                    _verify_checkpoint(tmp_path, expected_sha)
                    os.replace(str(tmp_path), str(final_path))
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise

        return str(clf_path), str(reg_path)

    @staticmethod
    def _get_device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"


def _log_mean_exp(log_values: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute log(mean(exp(log_values))) using numerically stable log-sum-exp.

    This prevents numerical underflow when probabilities are very small.

    Args:
        log_values: Array of log probabilities, shape (n_permutations, n_samples)
        axis: Axis along which to average (default 0, across permutations)

    Returns:
        Array of log-averaged probabilities, shape (n_samples,)
    """
    max_val = np.max(log_values, axis=axis, keepdims=True)
    # Subtract max for numerical stability: exp(log_p - max) is safer
    exp_vals = np.exp(log_values - max_val)
    mean_exp = np.mean(exp_vals, axis=axis)
    # Add back the max: log(mean(exp(log_p - max))) + max = log(mean(exp(log_p)))
    return np.log(mean_exp + 1e-10) + np.squeeze(max_val, axis=axis)


def _efficient_random_permutation(
    indices: list[int],
    n_permutations: int = 10,
    seed: int = 0,
) -> List[Tuple[int, ...]]:
    """
    Copy from: https://github.com/PriorLabs/tabpfn-extensions/blob/a54bc14398a5155ae22a5c0ac2fb9327e88782a8/src/tabpfn_extensions/unsupervised/unsupervised.py#L888-L911
    Generate multiple unique random permutations of the given indices.

    Args:
        indices: List of indices to permute
        n_permutations: Number of unique permutations to generate
        seed: Random seed

    Returns:
        List of unique permutations
    """
    perms = []
    seen = set()
    n_iter = 0
    max_iterations = n_permutations * 10  # Set a limit to avoid infinite loops

    while len(perms) < n_permutations and n_iter < max_iterations:
        perm = _efficient_random_permutation_(indices, seed + n_iter)
        if perm not in seen:
            perms.append(perm)
            seen.add(perm)
        n_iter += 1

    if len(perms) < n_permutations:
        import math
        n_possible = math.factorial(len(indices))
        logging.getLogger(__name__).warning(
            f"Requested {n_permutations} unique permutations of {len(indices)} features, "
            f"but only {len(perms)} could be generated ({n_possible} possible). "
            f"Outlier score estimates may have higher variance."
        )

    return perms


def _efficient_random_permutation_(
    indices: list[int],
    seed: int = 0,
) -> Tuple[int, ...]:
    """
    Copy from https://github.com/PriorLabs/tabpfn-extensions/blob/a54bc14398a5155ae22a5c0ac2fb9327e88782a8/src/tabpfn_extensions/unsupervised/unsupervised.py#L914-L933
    Generate a single random permutation from the given indices.

    Args:
        indices: List of indices to permute
        seed: Random seed

    Returns:
        A tuple representing a random permutation of the input indices
    """
    rng = random.Random(seed)
    # Create a copy of the list to avoid modifying the original
    permutation = list(indices)

    # Shuffle the list in-place using Fisher-Yates algorithm
    for i in range(len(indices) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = rng.randint(0, i)
        # Swap elements at i and j
        permutation[i], permutation[j] = permutation[j], permutation[i]

    return tuple(permutation)


def _to_numeric_fast(x: np.ndarray) -> np.ndarray:
    """
    Fast conversion for numpy arrays, used in density estimation.
    Assumes x is already numeric or will be coerced.
    """
    try:
        return np.asarray(x, dtype=np.float32)
    except (ValueError, TypeError):
        # Fallback: try to encode strings as integers
        from sklearn.preprocessing import LabelEncoder
        result = np.zeros_like(x, dtype=np.float32)
        for col_idx in range(x.shape[1] if len(x.shape) > 1 else 1):
            col = x if len(x.shape) == 1 else x[:, col_idx]
            if col.dtype.kind in ('U', 'S', 'O'):
                le = LabelEncoder()
                result[:, col_idx] = le.fit_transform(col.astype(str))
            else:
                result[:, col_idx] = col.astype(np.float32)
        return result


def _to_numeric(x: dt.Frame, ord_encoder: OrdinalEncoder,
                numeric_col_mask: Optional[np.ndarray] = None) -> np.ndarray:
    assert len(x.shape) == 2

    if x.shape[0] == 0:
        return x.to_numpy().astype(np.float32)

    if numeric_col_mask is None:
        numeric_col_mask = _numeric_column_mask(x)
    if np.all(numeric_col_mask):
        return x.to_numpy().astype(np.float32)

    numeric_col_indices = np.where(numeric_col_mask)[0].tolist()
    non_numeric_col_indices = np.where(~numeric_col_mask)[0].tolist()

    numeric_array = np.empty((x.nrows, x.ncols), dtype=np.float32)

    if numeric_col_indices:
        numeric_array[:, numeric_col_indices] = x[:, numeric_col_indices].to_numpy().astype(np.float32)

    if non_numeric_col_indices:
        x_cat = x[:, non_numeric_col_indices].to_numpy().astype(np.object_)
    else:
        return numeric_array

    try:
        check_is_fitted(ord_encoder, attributes=["categories_"])
        numeric_transformed = ord_encoder.transform(x_cat)
    except NotFittedError:
        numeric_transformed = ord_encoder.fit_transform(x_cat)

    numeric_array[:, non_numeric_col_indices] = np.asarray(numeric_transformed, dtype=np.float32)

    del x_cat, numeric_transformed

    return numeric_array


def _numeric_column_mask(x: dt.Frame) -> np.ndarray:
    mask = np.zeros(x.shape[1], dtype=bool)

    # Pre-compute shared sample indices once for all string columns
    sample_size = min(10000, x.nrows)
    if sample_size < x.nrows:
        rng = np.random.RandomState(42)
        shared_indices = rng.choice(x.nrows, size=sample_size, replace=False)
        shared_indices.sort()
        shared_indices = shared_indices.tolist()
    else:
        shared_indices = None

    for i in range(x.shape[1]):
        col_type = x[:, i].type

        if col_type in (dt.Type.int8, dt.Type.int16, dt.Type.int32, dt.Type.int64,
                       dt.Type.float32, dt.Type.float64, dt.Type.bool8):
            mask[i] = True
        else:
            # For string columns, sample rows to check if they're numeric-like.
            # Use random sampling to avoid positional bias (e.g., first rows are IDs).
            if shared_indices is not None:
                sample = x[shared_indices, i].to_numpy()
            else:
                sample = x[:, i].to_numpy()
            try:
                # Filter out None/NaN so numeric columns with missing values
                # are not incorrectly classified as non-numeric.
                sample_clean = sample[sample != None]  # noqa: E711
                if len(sample_clean) > 0:
                    sample_clean.astype(np.float32)
                    mask[i] = True
                else:
                    mask[i] = False
            except (ValueError, TypeError):
                mask[i] = False

    return mask


def _get_cache_dir() -> pathlib.Path:
    return pathlib.Path(systemutils.temporary_files_abspath) / "tabpfn_cache"


def _verify_checkpoint(path: pathlib.Path, expected_sha256: Optional[str]) -> None:
    """Verify SHA-256 checksum of a downloaded checkpoint file.

    Skipped when expected_sha256 is None (hashes not yet populated).
    Raises RuntimeError on mismatch; caller is responsible for cleanup.
    """
    if expected_sha256 is None:
        return
    import hashlib
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    if sha256.hexdigest() != expected_sha256:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: "
            f"expected {expected_sha256}, got {sha256.hexdigest()}."
        )
