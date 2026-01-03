"""
TabPFN-based outlier scoring transformer for Driverless AI.

License compliance note (Prior Labs License v1.2, See license text; ensure compliance with attribution requirements):
- This recipe depends on `tabpfn` / `tabpfn-extensions` and may download/use TabPFN weights.
- If you DISTRIBUTE or make available a product/service containing TabPFN source/weights (or derivative work),
    you must satisfy the license additional attribution requirement (Section 10), including prominently displaying:
    “Built with PriorLabs-TabPFN” in relevant UI/docs.
"""
import logging
import math
import os
import pathlib
import random
import time
from typing import Any
from typing import ClassVar
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import Union

import datatable as dt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.utils.multiclass import unique_labels

from h2oaicore import systemutils
from h2oaicore.models import CustomModel
from h2oaicore.systemutils_more import download


TABPFN_CLASSIFIER_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
)
TABPFN_REGRESSOR_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-regressor.ckpt"
)


class InferenceFunction(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


class UtilityFunction(Protocol):
    def __call__(
        self,
        predicted: np.ndarray,
        actual: Optional[np.ndarray] = None,
        labels: Optional[Sequence[int]] = None,
    ) -> Union[float, np.ndarray]:
        ...


class MarginalImputer:
    """Marginalizing out removed features with their marginal distribution."""

    def __init__(self, model_callable: InferenceFunction, data: np.ndarray):
        self.model_callable = model_callable
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        self.num_groups = data.shape[1]

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Prepare x and S.
        assert x.shape[0] == mask.shape[0]
        n = len(x)
        x = x.repeat(self.samples, 0)
        mask = mask.repeat(self.samples, 0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~mask] = self.data_repeat[~mask]

        # Make predictions.
        pred = self.model_callable(x_)
        if pred.ndim == 2 and pred.shape[1] == 2:
            pred = pred[:, 1]
            pred = pred.reshape(-1, self.samples)
        else:
            pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


class PermutationEstimator:
    """Inspired from https://github.com/iancovert/sage/blob/master/sage"""
    def __init__(
        self,
        imputer: MarginalImputer,
        utility_fn: UtilityFunction,
        n_jobs: int = 1,
        random_state: int = 1234,
        num_classes: int = 1,
    ):
        self.imputer = imputer
        self.utility_fn = utility_fn
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.num_classes = num_classes

    def __call__(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        batch_size: int = 512,
        thresh: float = 0.95,
        n_permutations: int = 8,
        min_coalition: float = 0.0,
        max_coalition: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        size, _ = X.shape
        num_features = self.imputer.num_groups

        # Determine min/max coalition sizes.
        if isinstance(min_coalition, float):
            min_coalition = int(min_coalition * num_features)
        if isinstance(max_coalition, float):
            max_coalition = int(max_coalition * num_features)
        assert min_coalition >= 0
        assert max_coalition <= num_features
        assert min_coalition < max_coalition

        if Y is None:
            systemutils.loggerinfo(logger, "Computing Shapley values")
            return self._process_local_explanation(
                x=X,
                batch_size=batch_size,
                n_permutations=n_permutations,
                min_coalition=min_coalition,
                max_coalition=max_coalition,
                logger=logger,
            )
        else:
            assert 0 < thresh < 1
            systemutils.loggerinfo(logger, "Computing Global Feature Importance through SAGE simulation")
            return self._process_global_explanation(
                x=X,
                y=Y,
                batch_size=batch_size,
                thresh=thresh,
                n_permutations=n_permutations,
                min_coalition=min_coalition,
                max_coalition=max_coalition,
                logger=logger,
            )

    def _process_local_explanation(
        self,
        x: np.ndarray,
        batch_size: int,
        n_permutations: int,
        min_coalition: int,
        max_coalition: int,
        logger: Optional[logging.Logger] = None,
    ) -> np.ndarray:
        size, _ = x.shape
        num_features = self.imputer.num_groups
        consider_class_idx = self.num_classes > 2
        n_loops = n_permutations

        if consider_class_idx:
            mean_samples = np.zeros((size, self.num_classes, num_features), dtype=np.float32)
        else:
            mean_samples = np.zeros((size, num_features), dtype=np.float32)

        for it in range(n_loops):
            batches = []
            for idx in range(int(np.ceil(size / batch_size))):
                stop = min((idx + 1) * batch_size, size)
                indices = np.arange(idx * batch_size, stop)
                batches.append((x[indices], indices))

            early_st = time.perf_counter()
            for idx, _batch in enumerate(batches):
                st = time.perf_counter()
                systemutils.loggerinfo(logger, f"Process {idx+1}/{len(batches)} batch for Shapley samples at {it + 1}/{n_loops} iteration with {_batch[0].shape[0]} samples")
                scores, _, indices = self._process_sample(
                    x=_batch[0],
                    y=None,
                    num_features=num_features,
                    max_coalition=max_coalition,
                    min_coalition=min_coalition,
                    seed=self.random_state + 10000 * it + idx,
                    consider_class_idx=consider_class_idx,
                    indices=_batch[1],
                )
                systemutils.loggerinfo(
                    logger,
                    f"Process {idx+1}/{len(batches)} batch for Shapley samples at {it + 1}/{n_loops} iteration took {(time.perf_counter() - st):.6f} seconds"
                )
                mean_samples[indices] += scores

            systemutils.loggerinfo(
                logger,
                f"Process batch for Shapley samples at {it + 1}/{n_loops} iteration took {(time.perf_counter() - early_st):.6f} seconds"
            )

        return (mean_samples / n_loops).astype(np.float32)

    def _process_global_explanation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        n_permutations: int,
        min_coalition: int,
        max_coalition: int,
        thresh: float,
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        size, _ = x.shape
        num_features = self.imputer.num_groups
        n_loops = int(np.ceil(n_permutations / (batch_size * self.n_jobs)))
        total_samples = np.zeros(num_features, dtype=np.int32)
        mean_samples = np.zeros(num_features, dtype=np.float32)
        M2 = np.zeros(num_features, dtype=np.float32)
        std = np.zeros(num_features, dtype=np.float32)

        rng = np.random.RandomState(self.random_state)
        for it in range(n_loops):
            batches = []
            for _ in range(self.n_jobs):
                indices = rng.choice(size, batch_size)
                batches.append((x[indices], y[indices]))

            early_st = time.perf_counter()
            for idx, _batch in enumerate(batches):
                st = time.perf_counter()
                systemutils.loggerinfo(logger, f"Processing batch {idx+1}/{len(batches)} batch for SAGE samples at {it + 1}/{n_loops} iteration with {_batch[0].shape[0]} samples")
                scores, sample_counts, _ = self._process_sample(
                    x=_batch[0],
                    y=_batch[1],
                    num_features=num_features,
                    min_coalition=min_coalition,
                    max_coalition=max_coalition,
                    seed=self.random_state + 10000 * it + idx,
                )
                systemutils.loggerinfo(
                    logger,
                    f"Process {idx+1}/{len(batches)} batch for SAGE samples at {it + 1}/{n_loops} iteration took {(time.perf_counter() - st):.6f} seconds")
                if sample_counts is None:
                    c = np.full(num_features, scores.shape[0], dtype=np.int64)
                else:
                    c = sample_counts.astype(np.int64, copy=False)

                # sum and sumsq over the batch (zeros for non-sampled features are OK if we use c)
                batch_sum = scores.sum(axis=0, dtype=np.float64)
                batch_sumsq = (scores.astype(np.float64) ** 2).sum(axis=0)
                batch_mean = np.zeros(num_features, dtype=np.float64)
                nz = c > 0
                batch_mean[nz] = batch_sum[nz] / c[nz]
                batch_M2 = np.zeros(num_features, dtype=np.float64)
                batch_M2[nz] = batch_sumsq[nz] - c[nz] * (batch_mean[nz] ** 2)

                # merge (Welford parallel merge), elementwise
                total_samples_old = total_samples.copy()
                total_samples_new = total_samples_old + c

                delta = batch_mean - mean_samples
                mean_samples[nz] += delta[nz] * (c[nz] / total_samples_new[nz])
                M2[nz] += batch_M2[nz] + (delta[nz] ** 2) * (total_samples_old[nz] * c[nz] / total_samples_new[nz])
                total_samples = total_samples_new

            systemutils.loggerinfo(
                logger,
                f"Process batch for SAGE samples at {it + 1}/{n_loops} iteration took {(time.perf_counter() - early_st):.6f} seconds"
            )

            valid = total_samples > 1
            std[valid] = np.sqrt(M2[valid] / (total_samples[valid] - 1))

            # Calculate progress.
            std_max = float(np.nanmax(std))
            gap = float(max(mean_samples.max() - mean_samples.min(), 1e-12))
            ratio = 1 - std_max / (std_max + gap)

            # Check for convergence.
            systemutils.loggerinfo(logger, f"Converge ratio is {ratio}, expect {thresh}")
            if ratio >= thresh:
                break

        return mean_samples, std

    def _process_sample(
        self,
        x: np.ndarray,
        num_features: int,
        min_coalition: int,
        max_coalition: int,
        seed: int,
        y: Optional[np.ndarray] = None,
        consider_class_idx: bool = False,
        indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Setup.
        rng = np.random.RandomState(seed)
        batch_size = len(x)
        batch_range = np.arange(batch_size)
        mask = np.zeros((batch_size, num_features), dtype=bool)
        permutations = np.tile(np.arange(num_features), (batch_size, 1))
        if consider_class_idx:
            scores = np.zeros((batch_size, self.num_classes, num_features))
        else:
            scores = np.zeros((batch_size, num_features))

        # Sample permutations.
        for i in range(batch_size):
            rng.shuffle(permutations[i])

        # Calculate sample counts.
        if min_coalition > 0 or max_coalition < num_features:
            sample_counts = np.zeros(num_features, dtype=int)
            for i in range(batch_size):
                sample_counts[permutations[i, min_coalition:max_coalition]] += 1
        else:
            sample_counts = None

        # Add necessary features to minimum coalition.
        for i in range(min_coalition):
            # Add next feature.
            index = permutations[:, i]
            mask[batch_range, index] = True

        # Make prediction with minimum coalition.
        y_hat = self.imputer(x, mask)
        prev_score = self.utility_fn(y_hat, y)

        _claim_memory()
        # Add all remaining features.
        for i in range(min_coalition, max_coalition):
            # Add next feature.
            index = permutations[:, i]
            mask[batch_range, index] = True

            # Make prediction with missing features.
            y_hat = self.imputer(x, mask)
            score = self.utility_fn(y_hat, y)

            # Calculate delta sample.
            if consider_class_idx:
                scores[batch_range, :, index] = prev_score - score
            else:
                scores[batch_range, index] = prev_score - score
            prev_score = score

        _claim_memory(force=True)
        return scores, sample_counts, indices


class TabPFNManyClassifier(BaseEstimator, ClassifierMixin):
    """
    Simplified implementation based on:
    https://github.com/PriorLabs/tabpfn-extensions/blob/1960cc63a419f9022e902c17bb2c5407ed9e5431/src/tabpfn_extensions/many_class/many_class_classifier.py#L33
    """
    _required_parameters: ClassVar[list[str]] = ["estimator"]

    def __init__(
        self,
        estimator: Any,
        *,
        alphabet_size: Optional[int] = None,
        n_estimators: Optional[int] = None,
        n_estimators_redundancy: int = 4,
        random_state: Optional[int] = None,
        labels: Optional[Sequence[Any]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.alphabet_size = alphabet_size
        self.n_estimators = n_estimators
        self.n_estimators_redundancy = n_estimators_redundancy
        self.labels = labels
        self.logger = logger

        # Learning outcome
        self.alphabet_size_: Optional[int] = None
        self.no_mapping_needed_: bool = False
        self.classes_: Optional[np.ndarray] = None
        self.estimators_: Sequence[Any] = []
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Sequence[str] = []
        self.x_train_: Optional[np.ndarray] = None
        self.y_train_per_estimator_: Optional[np.ndarray] = None
        self.code_book_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "TabPFNManyClassifier":
        """Prepare classifier using custom validate_data.
        Actual fitting of sub-estimators happens in predict_proba if mapping is needed.
        """
        # Use the custom validate_data for y
        # Assuming it handles conversion to 1D and basic checks.
        # y_numeric=True is common for classification targets.
        x, y = self.validate_data(
            estimator=self,
            X=X,
            y=y,
            reset=True,
            force_all_finite=False,  # scikit-learn sets self.n_features_in_ automatically
        )
        self.n_features_in_ = x.shape[1]

        feature_names = fit_params.get("feature_names", [])
        if feature_names:
            self.feature_names_in_ = feature_names

        random_state_instance = check_random_state(self.random_state)
        self.classes_ = np.asarray(self.labels) if self.labels is not None else unique_labels(y)

        if self.labels is not None:
            missing = set(np.unique(y)) - set(self.classes_)
            if missing:
                raise ValueError(f"y contains labels not in `labels`: {sorted(list(missing))[:10]}")

        n_classes = len(self.classes_)
        self.alphabet_size_ = self._get_alphabet_size()
        self.no_mapping_needed_ = n_classes <= self.alphabet_size_

        if n_classes == 0:
            raise ValueError("Cannot fit with no classes present.")
        if n_classes == 1:
            # Gracefully handle single-class case: fit estimator, set trivial codebook
            cloned_estimator = clone(self.estimator)
            cloned_estimator.fit(x, y, **fit_params)
            self.no_mapping_needed_ = True
            self.estimators_ = [cloned_estimator]
            self.code_book_ = np.zeros((1, 1), dtype=int)
            return self

        if self.no_mapping_needed_:
            cloned_estimator = clone(self.estimator)
            # Base estimator fits on X_validated (already processed by custom validate_data)
            cloned_estimator.fit(x, y, **fit_params)
            self.estimators_ = [cloned_estimator]
            # Ensure n_features_in_ matches the fitted estimator if it has the attribute
            if hasattr(cloned_estimator, "n_features_in_"):
                self.n_features_in_ = cloned_estimator.n_features_in_
        else:
            n_est = self._get_n_estimators(n_classes, self.alphabet_size_)
            self.code_book_ = self._generate_codebook(
                n_classes, n_est, self.alphabet_size_, random_state_instance
            )
            classes_index_ = {c: i for i, c in enumerate(self.classes_)}
            self.x_train_ = x  # Store validated x
            y_indices = np.array([classes_index_[val] for val in y])
            self.y_train_per_estimator_ = self.code_book_[:, y_indices]

            # Pre-fit all sub-estimators to make prediction faster
            self.estimators_ = []
            for i in range(self.code_book_.shape[0]):
                est = clone(self.estimator)
                est.fit(x, self.y_train_per_estimator_[i, :], **fit_params)
                self.estimators_.append(est)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict multi-class targets for X."""
        # Attributes to check if fitted, adapt from user's ["_tree", "X", "y"]
        check_is_fitted(self, ["classes_", "n_features_in_"])
        # X will be validated by predict_proba or base_estimator.predict

        if self.no_mapping_needed_ or (
            hasattr(self, "estimators_")
            and self.estimators_ is not None
            and len(self.estimators_) == 1
        ):
            if not self.estimators_:
                raise RuntimeError("Estimator not fitted. Call fit first.")
            # Base estimator's predict validates X
            return self.estimators_[0].predict(X)

        predictions = self.predict_proba(X)
        if predictions.shape[0] == 0:
            return np.array([], dtype=self.classes_.dtype)
        return self.classes_[np.argmax(predictions, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for X. Sub-estimators are fitted here if mapping is used."""
        # Attributes to check if fitted, adapt from user's ["_tree", "X", "y"]
        # Key attributes for this classifier: classes_ must be set, n_features_in_ for X dim check.
        check_is_fitted(self, ["classes_", "n_features_in_"])

        # Use the custom validate_data for X in predict methods as well
        # reset=False as n_features_in_ should already be set from fit
        # Align DataFrame columns if needed
        x = self.validate_data(
            estimator=self,
            X=X,
            reset=False,
            force_all_finite=False,
        )

        if self.no_mapping_needed_:
            if not self.estimators_:
                raise RuntimeError("Estimator not fitted. Call fit first.")
            return self.estimators_[0].predict_proba(x)

        if (
                self.x_train_ is None
                or self.y_train_per_estimator_ is None
                or self.code_book_ is None
        ):
            raise RuntimeError(
                "Fit method did not properly initialize for mapping. Call fit first."
            )

        n_samples = x.shape[0]
        n_orig_classes = len(self.classes_)
        rest_class_code = self.alphabet_size_ - 1

        raw_probabilities = np.zeros((n_samples, n_orig_classes))
        counts = np.zeros(n_orig_classes, dtype=float)
        for i, est in enumerate(self.estimators_):
            proba = est.predict_proba(x)

            cols = np.asarray(est.classes_, dtype=int)
            if np.any(cols < 0) or np.any(cols >= self.alphabet_size_):
                raise ValueError("Base estimator produced invalid code labels.")
            if np.unique(cols).size != cols.size:
                raise ValueError("Duplicate class labels in base estimator.")

            full_i = np.zeros((n_samples, self.alphabet_size_), dtype=np.float64)
            full_i[:, cols] = proba

            codes_i = self.code_book_[i]
            valid = codes_i != rest_class_code
            raw_probabilities[:, valid] += full_i[:, codes_i[valid]]
            counts[valid] += 1.0

        valid_counts_mask = counts > 0
        final_probabilities = np.zeros_like(raw_probabilities)
        if np.any(valid_counts_mask):
            final_probabilities[:, valid_counts_mask] = (
                    raw_probabilities[:, valid_counts_mask] / counts[valid_counts_mask]
            )
        if not np.all(valid_counts_mask):
            systemutils.loggerwarning(
                self.logger,
                "Some classes had zero specific code assignments during aggregation.",
            )

        prob_sum = np.sum(final_probabilities, axis=1, keepdims=True)
        safe_sum = np.where(prob_sum < 1e-8, 1.0, prob_sum)
        # Normalization
        final_probabilities /= safe_sum
        final_probabilities[prob_sum.squeeze() < 1e-8] = 1.0 / n_orig_classes

        return final_probabilities

    @staticmethod
    def validate_data(
        estimator: BaseEstimator,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        reset: bool = False,
        validate_separately: bool = False,
        force_all_finite: bool = False,
        **kwargs
    ):
        """
        Validate input data and set or check feature names and counts of the input.

        See the original scikit-learn documentation:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.validate_data.html#sklearn.utils.validation.validate_data
        """
        return estimator._validate_data(
            X=X if X is not None else "no_validation",
            y=y if y is not None else "no_validation",
            reset=reset,
            validate_separately=validate_separately,
            force_all_finite=force_all_finite,
            **kwargs
        )

    @staticmethod
    def _generate_codebook(
        n_classes: int,
        n_estimators: int,
        alphabet_size: int,
        random_state_instance: np.random.RandomState,
    ) -> np.ndarray:
        """Generate codebook with balanced coverage."""
        if n_classes <= alphabet_size:
            raise ValueError(
                "_generate_codebook called when n_classes <= alphabet_size"
            )

        codes_to_assign = list(range(alphabet_size - 1))
        n_codes_available = len(codes_to_assign)
        rest_class_code = alphabet_size - 1

        if n_codes_available == 0:
            raise ValueError(
                "alphabet_size must be at least 2 for codebook generation."
            )

        codebook = np.full((n_estimators, n_classes), rest_class_code, dtype=int)
        coverage_count = np.zeros(n_classes, dtype=int)

        for i in range(n_estimators):
            n_assignable_this_row = min(n_codes_available, n_classes)
            noisy_counts = coverage_count + random_state_instance.uniform(
                0, 0.1, n_classes
            )
            sorted_indices = np.argsort(noisy_counts)
            selected_classes_for_row = sorted_indices[:n_assignable_this_row]
            permuted_codes = random_state_instance.permutation(codes_to_assign)
            codes_to_use = permuted_codes[:n_assignable_this_row]
            codebook[i, selected_classes_for_row] = codes_to_use
            coverage_count[selected_classes_for_row] += 1

        if np.any(coverage_count == 0):
            uncovered_indices = np.where(coverage_count == 0)[0]
            raise RuntimeError(
                f"Failed to cover classes within {n_estimators} estimators. "
                f"{len(uncovered_indices)} uncovered (e.g., {uncovered_indices[:5]}). "
                f"Increase `n_estimators` or `n_estimators_redundancy`."
            )

        return codebook

    def _get_alphabet_size(self) -> int:
        """Helper to get alphabet_size, inferring if necessary."""
        if self.alphabet_size_ is not None:
            return self.alphabet_size_
        if self.alphabet_size is not None:
            return self.alphabet_size
        try:
            # TabPFN specific attribute, or common one for models with class limits
            return self.estimator.max_num_classes_
        except AttributeError:
            # Fallback for estimators not exposing this directly
            # Might need to be explicitly set for such estimators.
            systemutils.loggerwarning(
                self.logger,
                "Could not infer alphabet_size from estimator.max_num_classes_."
                " Ensure alphabet_size is correctly set if this is not TabPFN.",
            )
            # Default to a common small number if not TabPFN and not set,
            # though this might not be optimal.
            return 10

    def _get_n_estimators(self, n_classes: int, alphabet_size: int) -> int:
        """Helper to calculate the number of estimators."""
        if self.n_estimators is not None:
            return self.n_estimators
        if n_classes <= alphabet_size:
            return 1  # Only one base estimator needed
        # Using max(2,...) ensures alphabet_size for log is at least 2
        min_estimators_theory = math.ceil(math.log(n_classes, max(2, alphabet_size)))
        # Ensure enough estimators for potential coverage based on n_classes
        min_needed_for_potential_coverage = math.ceil(
            n_classes / max(1, alphabet_size - 1)
        )
        return (
            max(min_estimators_theory, min_needed_for_potential_coverage)
            * self.n_estimators_redundancy
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "classifier"
        return tags


class TabPFNModel(CustomModel):
    r"""
        TabPFN Model for Driverless AI.

        CAUTION: TabPFN pretrained model has fitting size limitation, the upperbound to be < 10000,
        > 10000 is technically feasible but performance can be jeopardized.
        GPU inference is highly recommend and this transformer works best to small dataset < 10000
        reference: https://github.com/PriorLabs/TabPFN

        **What it does**
        - Fits a supervised TabPFNRegressor/TabPFNManyClassifier on the transformed dataset.

        **References**
        - Upstream TabPFN utilities:
          https://github.com/PriorLabs/tabpfn-extensions/blob/1960cc63a419f9022e902c17bb2c5407ed9e5431/src/tabpfn_extensions/many_class/many_class_classifier.py
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
        2) prominently display: **“Built with PriorLabs-TabPFN”** on relevant UI/docs pages.
        (Internal benchmarking/testing without external communication does not qualify as distribution per the license.)
    """
    _supports_sample_weight = False
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "TabPFNModel"
    _description = "TabPFN Model capable of fitting regression and classification models"
    _testing_can_skip_failure = False
    _is_reproducible = True
    _fit_by_iteration = True
    _fit_iteration_name = "n_estimators"
    _modules_needed_by_name = ["tabpfn==6.2.0"]
    _can_use_gpu = True
    _can_use_multi_gpu = False
    _get_gpu_lock = True
    _get_gpu_lock_vis = True
    _must_use_gpu = True

    MAX_CLASSES = 10
    TRAIN_SIZE_LIMITS = 10000
    TRAIN_SIZE_OVERLOAD_RATE = 2
    MAX_FEATURES = 20 # very sensitive to SAGE/Shapley O(#features * #batch * #permutations * O(fit/predict)), reduce the value if too slow
    MAX_GLOBAL_EXPLANATION_PERMUTATIONS = 512
    MAX_LOCAL_EXPLANATION_PERMUTATIONS = 12 # very sensitive to running complexity, pick small to be conservative
    FAST_LOCAL_EXPLANATION_PERMUTATIONS = 5
    MAX_CONTEXT_SIZE = 64 # very sensitive to GPU memory when enabled, should be chosen close to batch size
    BATCH_SIZE = 64 # very sensitive to GPU memory when enabled, pick small to be conservative

    @staticmethod
    def can_use(
        accuracy,
        interpretability,
        train_shape=None,
        test_shape=None,
        valid_shape=None,
        n_gpus=0,
        num_classes=None,
        **kwargs
    ):
        return (
            accuracy > 8
            and interpretability < 2
            and train_shape[0] < int(TabPFNModel.TRAIN_SIZE_OVERLOAD_RATE * TabPFNModel.TRAIN_SIZE_LIMITS)
            and train_shape[1] <= TabPFNModel.MAX_FEATURES
            and n_gpus > 0
        )

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    @property
    def has_pred_contribs(self):
        return True

    @property
    def has_output_margin(self):
        return True

    @property
    def is_classification(self) -> bool:
        return self.n_classes > 1

    @property
    def is_many_classification(self) -> bool:
        return self.n_classes > self.MAX_CLASSES

    @property
    def is_default_classification(self) -> bool:
        return self.is_classification and not self.is_many_classification

    @property
    def n_classes(self) -> int:
        return len(self.labels) if self.labels is not None else 1

    def set_default_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        if accuracy is None or accuracy < 5:
            self.params = dict(
                n_estimators=8,
                n_estimators_redundancy=4,
                softmax_temperature=0.9,
                balance_probabilities=False,
                average_before_softmax=False,
                tune_boundary_threshold=False,
                calibrate_softmax_temperature=False,
            )
        elif accuracy > 8:
            self.params = dict(
                n_estimators=12,
                n_estimators_redundancy=6,
                softmax_temperature=0.3,
                balance_probabilities=True,
                average_before_softmax=True,
                tune_boundary_threshold=False,
                calibrate_softmax_temperature=False,
            )
        else:
            self.params = dict(
                n_estimators=10,
                n_estimators_redundancy=5,
                softmax_temperature=0.6,
                balance_probabilities=True,
                average_before_softmax=True,
                tune_boundary_threshold=True,
                calibrate_softmax_temperature=True,
            )

    def mutate_params(self, accuracy=10, time_tolerance=10, interpretability=1, score_f_name: str = None, trial=None, **kwargs):
        self.params["balance_probabilities"] = False
        self.params["average_before_softmax"] = False
        self.params["tune_boundary_threshold"] = False
        self.params["calibrate_softmax_temperature"] = False
        # Consider increasing n_estimator only if higher model performance is required and GPU is powerful enough, otherwise very slow
        if accuracy > 8:
            n_estimator_list = [10, 12, 14]
            n_estimator_redundancy_list = [5, 6, 7]
            self.params["balance_probabilities"] = True
            self.params["average_before_softmax"] = True
            self.params["tune_boundary_threshold"] = True
            self.params["calibrate_softmax_temperature"] = True # Caution, expensive tuning
            max_softmax_temperature = 0.5
            min_softmax_temperature = 0.1
        elif accuracy > 4:
            n_estimator_list = [8, 10, 12]
            n_estimator_redundancy_list = [4, 5, 6]
            self.params["balance_probabilities"] = True
            self.params["average_before_softmax"] = True
            max_softmax_temperature = 0.8
            min_softmax_temperature = 0.5
        else:
            n_estimator_list = [6, 8, 10]
            n_estimator_redundancy_list = [3, 4, 5]
            max_softmax_temperature = 1.0
            min_softmax_temperature = 0.8

        self.params["n_estimators"] = int(random.choice(n_estimator_list))
        self.params["n_estimators_redundancy"] = random.choice(n_estimator_redundancy_list)
        self.params["softmax_temperature"] = random.choice(np.linspace(min_softmax_temperature, max_softmax_temperature))

    def fit(
        self, X: dt.Frame,
        y: np.array,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[Sequence[Tuple[dt.Frame, np.ndarray]]] = None,
        sample_weight_eval_set: Optional[Sequence[np.ndarray]] = None,
        **kwargs
    ) -> None:
        logger = None
        if self.context and self.context.experiment_id:
            logger = systemutils.make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        enc_labels = self._encode_labels()
        self._prepare_env(self.random_state)

        ord_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )

        y = self._prepare_y(y, self.is_classification)
        x_numpy, sample_indices = self._prepare_x(x=X, encoder=ord_encoder_, y=y)
        x_fit = x_numpy[sample_indices]
        device = self._get_device()
        n_jobs = self._get_n_jobs(logger, **kwargs)
        model = self._get_tabpfn_model(n_jobs=n_jobs, device=device, enc_labels=enc_labels, logger=logger)

        y_fit = y[sample_indices]
        st = time.perf_counter()
        model.fit(X=x_fit, y=y_fit)
        systemutils.loggerinfo(logger, f"[{self.__class__.__name__}] [fit] TabPFN Model fit takes {(time.perf_counter() - st):.6f} seconds")

        # Find background data points for local explanations
        # Good heuristic with a representative coreset, better than random sampling
        x_bg = _kmeans_snap_coreset(X=x_fit, k=min(self.MAX_CONTEXT_SIZE, len(x_fit)), random_state=self.random_state)

        feat_imp: Optional[np.ndarray] = None
        if eval_set is not None:
            _claim_memory()
            x_eval, y_eval = eval_set[0]
            y_val = self._prepare_y(y_eval, self.is_classification)
            x_val, _ = self._prepare_x(x=x_eval, encoder=ord_encoder_, y=y_val)
            if self.is_classification:
                utility_fn = _get_classification_utility()
            else:
                utility_fn = _get_regression_utility()
            imputer = MarginalImputer(
                model_callable=self._get_model_inference(model),
                data=x_bg,
            )
            estimator = PermutationEstimator(
                imputer=imputer,
                utility_fn=utility_fn,
                n_jobs=self._get_n_jobs(logger, **kwargs),
                random_state=self.random_state,
                num_classes=self.n_classes,
            )

            systemutils.loggerinfo(logger, "Start SAGE simulation game to compute feature importance...")
            st = time.perf_counter()
            feat_imp, _ = estimator(
                X=x_val,
                Y=y_val,
                batch_size=self.BATCH_SIZE,
                n_permutations=self.MAX_GLOBAL_EXPLANATION_PERMUTATIONS,
                logger=logger,
            )
            systemutils.loggerinfo(logger, f"Finished SAGE simulation game takes {(time.perf_counter() - st):.6f} seconds")
        else:
            systemutils.loggerwarning(logger, f"[{self.__class__.__name__}] [fit] Skip computing features global importance")

        _claim_memory(force=True)
        self.set_model_properties(model=(model, ord_encoder_, x_bg), features=X.names, importances=feat_imp)
        return None

    def predict(self, X, **kwargs):
        pred_contribs = kwargs.get('pred_contribs', False)
        fast_approx = kwargs.pop('fast_approx', False)

        logger = None
        if self.context and self.context.experiment_id:
            logger = systemutils.make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        model, _, _, _ = self.get_model_properties()
        fitted_model, ord_encoder, x_bg = model

        self._prepare_env(self.random_state)
        x_predict, _ = self._prepare_x(x=X, encoder=ord_encoder)

        st = time.perf_counter()
        if self.is_classification:
            predictions = fitted_model.predict_proba(x_predict)
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]
        else:
            predictions = np.asarray(fitted_model.predict(x_predict), dtype=np.float32)
        systemutils.loggerinfo(
            logger,
            f"[{self.__class__.__name__}] [predict] TabPFN Model prediction takes {(time.perf_counter() - st):.6f} seconds",
        )

        if not pred_contribs:
            return predictions
        else:
            _claim_memory()
            imputer = MarginalImputer(
                model_callable=self._get_model_inference(fitted_model, logits=True),
                data=x_bg,
            )
            estimator = PermutationEstimator(
                imputer=imputer,
                utility_fn=_get_prediction_utility(),
                n_jobs=self._get_n_jobs(logger, **kwargs),
                random_state=self.random_state,
                num_classes=self.n_classes,
            )

            systemutils.loggerinfo(logger, f"Start {'Fast' if fast_approx else ''} Shapley computation...")
            st = time.perf_counter()
            phi = estimator(
                X=x_predict,
                batch_size=self.BATCH_SIZE,
                n_permutations=self.FAST_LOCAL_EXPLANATION_PERMUTATIONS if fast_approx else self.MAX_LOCAL_EXPLANATION_PERMUTATIONS,
                logger=logger,
            )
            systemutils.loggerinfo(logger, f"Finished {'Fast' if fast_approx else ''} Shapley computation takes {(time.perf_counter() - st):.6f} seconds")
            _claim_memory(force=True)

            bias = predictions - phi.sum(axis=-1)
            bias = np.expand_dims(bias, axis=-1)
            pred_shap = np.concatenate((phi, bias), axis=-1).astype(np.float32)
            if pred_shap.ndim == 3:
                pred_shap = pred_shap.reshape(pred_shap.shape[0], -1)
            return pred_shap

    def _prepare_x(self, x: dt.Frame, encoder: OrdinalEncoder, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        sample_indices = np.arange(x.shape[0])
        n = x.shape[0]

        if n > self.TRAIN_SIZE_LIMITS:
            if y is not None and self.is_classification:
                sample_indices, _ = train_test_split(
                    np.arange(n),
                    train_size=self.TRAIN_SIZE_LIMITS,
                    stratify=y,
                    random_state=self.random_state,
                )
            else:
                rng = np.random.RandomState(self.random_state)
                sample_indices = rng.choice(n, size=self.TRAIN_SIZE_LIMITS, replace=False)

        x_numpy = _to_numeric(x, encoder)
        return np.asarray(x_numpy, dtype=np.float32), sample_indices

    def _get_tabpfn_model(self, n_jobs: int, device: str, enc_labels: Optional[Sequence[int]], logger: Optional[logging.Logger] = None):
        n_estimators = self.params.get("n_estimators", 8)
        softmax_temperature = self.params.get("softmax_temperature", 0.9)
        balance_probabilities = self.params.get("balance_probabilities", False)
        average_before_softmax = self.params.get("average_before_softmax", False)
        tune_boundary_threshold = self.params.get("tune_boundary_threshold", False)
        calibrate_softmax_temperature = self.params.get("calibrate_softmax_temperature", False)
        n_estimators_redundancy = self.params.get("n_estimators_redundancy", 4)

        systemutils.loggerinfo(
            logger,
            f"parameters: n_estimators = {n_estimators}, softmax_temperature = {softmax_temperature}, balance_probabilities = {balance_probabilities},"
            f" average_before_softmax = {average_before_softmax}, tune_boundary_threshold = {tune_boundary_threshold},"
            f" calibrate_softmax_temperature = {calibrate_softmax_temperature}, n_estimators_redundancy = {n_estimators_redundancy}"
        )
        tabpfn_classifier, tabpfn_regressor = self._build_tabpfn_models(
            seed=self.random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            device=device,
            softmax_temperature=softmax_temperature,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            tune_boundary_threshold=tune_boundary_threshold,
            calibrate_softmax_temperature=calibrate_softmax_temperature,
            logger=logger,
        )

        if self.is_many_classification:
            return TabPFNManyClassifier(
                estimator=tabpfn_classifier,
                n_estimators=n_estimators,
                n_estimators_redundancy=n_estimators_redundancy,
                random_state=self.random_state,
                labels=[] if enc_labels is None else enc_labels,
                logger=logger,
            )
        elif self.is_classification:
            return tabpfn_classifier
        else:
            return tabpfn_regressor

    def _encode_labels(self) -> Optional[Sequence[int]]:
        if self.is_classification:
            assert self.labels is not None
            enc_labels = LabelEncoder().fit_transform(self.labels)
            return np.sort(np.unique(enc_labels))
        return self.labels

    def _get_model_inference(self, model, logits: bool = False) -> InferenceFunction:
        return ModelInferenceCallable(
            model=model,
            is_classification=self.is_classification,
            logits=logits,
        )

    @staticmethod
    def _build_tabpfn_models(
        seed: int,
        n_jobs: int,
        n_estimators: int,
        device: str,
        softmax_temperature: float = 0.5,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        tune_boundary_threshold: bool = False,
        calibrate_softmax_temperature: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Instantiate TabPFNClassifier/Regressor in a way compatible with your environment.
        You may need to adjust constructor args to point to ckpt paths depending on TabPFN version.
        """
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        clf_ckpt, reg_ckpt = TabPFNModel._ensure_weights_cached()

        systemutils.loggerinfo(logger, f"Instantiating TabPFN Classifier and Regressor. {clf_ckpt} and {reg_ckpt}")
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
            tuning_config={"tune_decision_thresholds": tune_boundary_threshold,
                           "calibrate_temperature": calibrate_softmax_temperature},
        )
        tabpfn_reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=device,
            model_path=reg_ckpt,
            random_state=seed,
            n_preprocessing_jobs=n_jobs,
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
        )

        return tabpfn_clf, tabpfn_reg

    @staticmethod
    def _ensure_weights_cached():
        """
        Optional: pre-download weights into a deterministic cache location.
        If TabPFN already handles caching, this still helps DAI environments.
        """
        cache_dir = _get_cache_dir()
        systemutils.makedirs(cache_dir, exist_ok=True)

        clf_path = cache_dir / os.path.basename(TABPFN_CLASSIFIER_CKPT_URL)
        reg_path = cache_dir / os.path.basename(TABPFN_REGRESSOR_CKPT_URL)

        if not clf_path.exists():
            download(TABPFN_CLASSIFIER_CKPT_URL, dest_path=cache_dir)
        if not reg_path.exists():
            download(TABPFN_REGRESSOR_CKPT_URL, dest_path=cache_dir)

        return str(clf_path), str(reg_path)

    @staticmethod
    def _get_device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _prepare_env(seed: int):
        import torch

        np.random.seed(seed)
        random.seed(seed)
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
        try:
            if systemutils.config.fixed_num_folds <= 0:
                n_jobs = max(1, int(
                    systemutils.max_threads() / min(systemutils.config.num_folds, kwargs['max_workers'])))
            else:
                n_jobs = max(1, int(
                    systemutils.max_threads() / min(systemutils.config.fixed_num_folds,
                                                    systemutils.config.num_folds, kwargs['max_workers'])))
        except KeyError:
            systemutils.loggerwarning(logger, "No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        return n_jobs

    @staticmethod
    def _prepare_y(y: np.ndarray, is_classification: bool) -> np.ndarray:
        transformed_y = y
        if transformed_y.dtype.kind in {"U", "S", "O"} or is_classification:
            le = LabelEncoder()
            transformed_y = le.fit_transform(transformed_y.astype(str))
            transformed_y = transformed_y.astype(np.int64, copy=False)
        return transformed_y


class PredictionUtility:
    """Picklable utility function for prediction-based Shapley values."""
    def __call__(self, predicted: np.ndarray, actual: Optional[np.ndarray] = None, labels: Optional[Sequence[int]] = None) -> Union[float, np.ndarray]:
        return -1 * predicted


class ClassificationUtility:
    """Picklable utility function for classification tasks using LogLoss."""
    def __init__(self):
        from h2oaicore import metrics
        self.scorer = metrics.LogLossScorer()

    def __call__(self, predicted: np.ndarray, actual: Optional[np.ndarray] = None, labels: Optional[Sequence[int]] = None) -> float:
        return self.scorer.score(actual=actual, predicted=predicted, labels=labels)


class RegressionUtility:
    """Picklable utility function for regression tasks using MSE."""
    def __init__(self):
        from h2oaicore import metrics
        self.scorer = metrics.MseScorer()

    def __call__(self, predicted: np.ndarray, actual: Optional[np.ndarray] = None, labels: Optional[Sequence[int]] = None) -> float:
        return self.scorer.score(actual=actual, predicted=predicted)


class ModelInferenceCallable:
    """Wrapper for model inference methods.

    Replaces lambda functions that capture model references.
    """
    def __init__(self, model, is_classification: bool, logits: bool = False):
        """
        Args:
            model: The fitted TabPFN model (Classifier/Regressor/ManyClassifier)
            is_classification: Whether this is a classification task
            logits: Whether to use predict_logits (currently unused)
        """
        self.model = model
        self.is_classification = is_classification
        self.logits = logits

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke model inference on input data."""
        if self.is_classification:
            return self.model.predict_proba(x)
        else:
            return self.model.predict(x)


def _get_prediction_utility() -> UtilityFunction:
    return PredictionUtility()


def _get_classification_utility() -> UtilityFunction:
    return ClassificationUtility()


def _get_regression_utility() -> UtilityFunction:
    return RegressionUtility()


def _kmeans_snap_coreset(X: np.ndarray, k: int, random_state: int = 0) -> np.ndarray:
    if k >= X.shape[0]:
        return X

    x = X
    if np.isnan(X).any():
        x = np.nan_to_num(X, nan=0.0)
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=2048, n_init="auto")
    km.fit(x)

    # snap each center to nearest actual row (medoid-like)
    labels = km.predict(x)
    idx = np.random.RandomState(random_state).choice(len(x), k, replace=False)
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        if cluster_mask.any():
            cluster_points = x[cluster_mask]
            # Find closest point to center
            distances = np.linalg.norm(cluster_points - km.cluster_centers_[cluster_id], axis=1)
            idx[cluster_id] = np.where(cluster_mask)[0][distances.argmin()]
    return x[idx].astype(np.float32)


def _to_numeric(x: dt.Frame, ord_encoder: OrdinalEncoder) -> np.ndarray:
    assert len(x.shape) == 2

    if x.shape[0] == 0:
        return x.to_numpy().astype(np.float32)

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

    for i in range(x.shape[1]):
        col_type = x[:, i].type

        if col_type in (dt.Type.int8, dt.Type.int16, dt.Type.int32, dt.Type.int64,
                       dt.Type.float32, dt.Type.float64, dt.Type.bool8):
            mask[i] = True
        else:
            # For string columns, check first 100 rows if they're numeric-like
            # This handles "123" stored as string without full column scan
            sample_size = min(100, x.nrows)
            sample = x[:sample_size, i].to_numpy()
            try:
                sample.astype(np.float32)
                mask[i] = True
            except (ValueError, TypeError):
                mask[i] = False

    return mask


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
                    gc.collect()
                return
            except (RuntimeError, AttributeError):
                # Fallback if memory stats unavailable
                torch.cuda.empty_cache()
        # CPU: collect garbage after all permutations for this chunk
    gc.collect()


def _get_cache_dir() -> pathlib.Path:
    return pathlib.Path(systemutils.temporary_files_abspath) / "tabpfn_cache"
