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
TABPFN_CLASSIFIER_CKPT_SHA256 = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
TABPFN_REGRESSOR_CKPT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"


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
        self._alloc_n = 1  # tracks how many repeats are currently allocated

    def __call__(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # Prepare x and S.
        assert x.shape[0] == mask.shape[0]
        n = len(x)
        x = x.repeat(self.samples, 0)
        mask = mask.repeat(self.samples, 0)

        # Prepare samples. Only re-allocate when batch size grows beyond
        # previous allocation; reuse (slice) the buffer for smaller batches.
        if n > self._alloc_n:
            self.data_repeat = np.tile(self.data, (n, 1))
            self._alloc_n = n
        needed = self.samples * n
        data_repeat = self.data_repeat[:needed]

        # Replace specified indices.
        x_ = x.copy()
        x_[~mask] = data_repeat[~mask]

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
        labels: Optional[Sequence[int]] = None,
    ):
        self.imputer = imputer
        self.utility_fn = utility_fn
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.num_classes = num_classes
        self.labels = [] if labels is None else list(labels)

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

        batches = []
        for idx in range(int(np.ceil(size / batch_size))):
            stop = min((idx + 1) * batch_size, size)
            indices = np.arange(idx * batch_size, stop)
            batches.append((x[indices], indices))

        wall_start = time.perf_counter()
        completed_loops = 0
        for it in range(n_loops):
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

            completed_loops += 1
            systemutils.loggerinfo(
                logger,
                f"Process batch for Shapley samples at {it + 1}/{n_loops} iteration took {(time.perf_counter() - early_st):.6f} seconds"
            )

        return (mean_samples / completed_loops).astype(np.float32)

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
        min_iterations = 2  # require at least 2 iterations to avoid premature convergence on degenerate data
        total_samples = np.zeros(num_features, dtype=np.int32)
        mean_samples = np.zeros(num_features, dtype=np.float64)
        M2 = np.zeros(num_features, dtype=np.float64)
        std = np.zeros(num_features, dtype=np.float64)

        rng = np.random.RandomState(self.random_state)
        wall_start = time.perf_counter()
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

            # Check for convergence (require minimum iterations to avoid
            # premature exit when std=0 on degenerate data).
            systemutils.loggerinfo(logger, f"Converge ratio is {ratio}, expect {thresh}")
            if it >= min_iterations - 1 and ratio >= thresh:
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
        prev_score = self.utility_fn(y_hat, y, self.labels)

        _claim_memory()
        # Add all remaining features.
        for i in range(min_coalition, max_coalition):
            # Add next feature.
            index = permutations[:, i]
            mask[batch_range, index] = True

            # Make prediction with missing features.
            y_hat = self.imputer(x, mask)
            score = self.utility_fn(y_hat, y, self.labels)

            # Calculate marginal contribution of adding feature `index`.
            # Sign convention: (prev_score - score) where score includes the
            # newly added feature. For PredictionUtility (local explanations),
            # utility = -1 * prediction, so:
            #   prev - curr = (-1*pred_before) - (-1*pred_after) = pred_after - pred_before
            # Positive delta means the feature increases the prediction.
            # This is consistent with standard Shapley attribution direction.
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
        self.mapping_fitted_: bool = False
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
            y_indices = np.array([classes_index_[val] for val in y])
            y_per_estimator = self.code_book_[:, y_indices]

            # Pre-fit all sub-estimators to make prediction faster
            self.estimators_ = []
            for i in range(self.code_book_.shape[0]):
                est = clone(self.estimator)
                est.fit(x, y_per_estimator[i, :], **fit_params)
                self.estimators_.append(est)
            self.mapping_fitted_ = True

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

        if not self.mapping_fitted_ or self.code_book_ is None:
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
        x_val = X if X is not None else "no_validation"
        y_val = y if y is not None else "no_validation"
        val_kwargs = dict(
            X=x_val, y=y_val, reset=reset,
            validate_separately=validate_separately,
            force_all_finite=force_all_finite, **kwargs,
        )
        try:
            # sklearn >= 1.6: standalone function
            from sklearn.utils.validation import validate_data as _sklearn_validate_data
            return _sklearn_validate_data(estimator, **val_kwargs)
        except ImportError:
            # sklearn < 1.6: deprecated instance method
            return estimator._validate_data(**val_kwargs)

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
    _modules_needed_by_name = ["tabpfn==6.4.1"]
    _can_use_gpu = True
    _can_use_multi_gpu = False
    _get_gpu_lock = True
    _get_gpu_lock_vis = True
    _must_use_gpu = True
    _can_handle_categorical = True
    _can_handle_non_numeric = True  # allow CatOriginalTransformer to pass raw categoricals
    _included_transformers = ["OriginalTransformer", "CatOriginalTransformer", "TextTransformer", "DatesTransformer"]
    #_is_reference = True        # reserved GA slot; final_model.py forces TabPFN into base-model list on top of ensemble_level
    #_num_allowed_reference = 1  # exactly 1 TabPFN slot in the GA — competes only with itself (hyperparameter search)
    #_ensemble_weight_floor = 0.0  # never pruned from stacked ensemble regardless of meta-learner weight

    MAX_CLASSES = 10
    TRAIN_SIZE_LIMITS = 10000
    TRAIN_SIZE_OVERLOAD_RATE = 2
    MAX_FEATURES = 500  # TabPFN V2 supports up to 500 natively; raw features (via _included_transformers) rarely exceed this
    MAX_GLOBAL_EXPLANATION_PERMUTATIONS = 256  # reduced from 1024 to control SAGE cost with higher feature counts
    MAX_LOCAL_EXPLANATION_PERMUTATIONS = 32 # very sensitive to running complexity, pick small to be conservative
    FAST_LOCAL_EXPLANATION_PERMUTATIONS = 10
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
            accuracy > 5
            and interpretability < 5
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
                finetune_epochs=10,
                finetune_learning_rate=1e-4,
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
                finetune_epochs=30,
                finetune_learning_rate=1e-5,
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
                finetune_epochs=20,
                finetune_learning_rate=5e-5,
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
            finetune_epochs_list = [25, 30, 40]
            finetune_lr_list = [5e-6, 1e-5, 2e-5]
            self.params["balance_probabilities"] = True
            self.params["average_before_softmax"] = True
            self.params["tune_boundary_threshold"] = True
            self.params["calibrate_softmax_temperature"] = True # Caution, expensive tuning
            max_softmax_temperature = 0.5
            min_softmax_temperature = 0.1
        elif accuracy > 4:
            n_estimator_list = [8, 10, 12]
            n_estimator_redundancy_list = [4, 5, 6]
            finetune_epochs_list = [15, 20, 25]
            finetune_lr_list = [2e-5, 5e-5, 1e-4]
            self.params["balance_probabilities"] = True
            self.params["average_before_softmax"] = True
            max_softmax_temperature = 0.8
            min_softmax_temperature = 0.5
        else:
            n_estimator_list = [6, 8, 10]
            n_estimator_redundancy_list = [3, 4, 5]
            finetune_epochs_list = [5, 10, 15]
            finetune_lr_list = [5e-5, 1e-4, 2e-4]
            max_softmax_temperature = 1.0
            min_softmax_temperature = 0.8

        rng = random.Random(self.random_state)
        self.params["n_estimators"] = int(rng.choice(n_estimator_list))
        self.params["n_estimators_redundancy"] = rng.choice(n_estimator_redundancy_list)
        self.params["softmax_temperature"] = rng.choice(np.linspace(min_softmax_temperature, max_softmax_temperature).tolist())
        self.params["finetune_epochs"] = int(rng.choice(finetune_epochs_list))
        self.params["finetune_learning_rate"] = rng.choice(finetune_lr_list)

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

        # self.labels empty or None indicates regression, convention in DAI.
        # Invalidate cached encoded labels in case labels changed between fits.
        if hasattr(self, '_cached_enc_labels'):
            del self._cached_enc_labels
        enc_labels = self._encode_labels()
        self._prepare_env(self.random_state)

        ord_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )

        y, y_label_encoder_ = self._prepare_y(y, self.is_classification)
        # Compute numeric column mask once during fit and reuse for eval_set + predict
        numeric_col_mask_ = _numeric_column_mask(X)
        # Pre-fit encoder on the FULL frame so it sees all categories,
        # even those in rows that _prepare_x will discard via subsampling.
        _fit_encoder(X, ord_encoder_, numeric_col_mask_)
        x_fit, sample_indices = self._prepare_x(x=X, encoder=ord_encoder_, y=y, logger=logger,
                                                numeric_col_mask=numeric_col_mask_)
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
        if eval_set is not None and len(eval_set) > 0 and x_fit.shape[1] <= 100:
            _claim_memory()
            x_eval, y_eval = eval_set[0]
            y_val, _ = self._prepare_y(y_eval, self.is_classification, label_encoder=y_label_encoder_)
            x_val, eval_sample_indices = self._prepare_x(x=x_eval, encoder=ord_encoder_, y=y_val, logger=logger,
                                                          numeric_col_mask=numeric_col_mask_)
            y_val = y_val[eval_sample_indices]
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
                labels=enc_labels,
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
        elif eval_set is not None and len(eval_set) > 0:
            # Skip SAGE for high-dimensional inputs (> 100 features) — use uniform importance
            feat_imp = np.ones(x_fit.shape[1], dtype=np.float32) / x_fit.shape[1]
            systemutils.loggerinfo(logger,
                f"[{self.__class__.__name__}] Skipping SAGE for {x_fit.shape[1]} features (> 100). "
                f"Using uniform importance.")
        else:
            systemutils.loggerwarning(logger, f"[{self.__class__.__name__}] [fit] Skip computing features global importance")

        _claim_memory(force=True)
        self.set_model_properties(model=(model, ord_encoder_, x_bg, numeric_col_mask_), features=X.names, importances=feat_imp)
        return None

    def predict(self, X, **kwargs):
        pred_contribs = kwargs.get('pred_contribs', False)
        output_margin = kwargs.get('output_margin', False)
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
        # Backward compat: older pickled models store 3-tuple, newer store 4-tuple
        if len(model) == 4:
            fitted_model, ord_encoder, x_bg, numeric_col_mask = model
        else:
            fitted_model, ord_encoder, x_bg = model
            numeric_col_mask = None

        self._prepare_env(self.random_state)
        x_predict, _ = self._prepare_x(x=X, encoder=ord_encoder, logger=logger,
                                        numeric_col_mask=numeric_col_mask)

        if not pred_contribs:
            st = time.perf_counter()
            if output_margin and self.is_classification:
                # Return logits in the same space that pred_contribs uses.
                # The Shapley path computes phi via _get_model_inference(logits=True)
                # and sets bias = logit(x) - sum(phi).  For the efficiency property
                # sum(contribs) == output_margin, this path must return the same
                # logit(x) value, cast to float32 to match pred_shap's dtype.
                logit_fn = self._get_model_inference(fitted_model, logits=True)
                predictions = np.asarray(logit_fn(x_predict), dtype=np.float32)
                if predictions.ndim == 2 and predictions.shape[1] == 2:
                    predictions = predictions[:, 1]
            elif self.is_classification:
                predictions = fitted_model.predict_proba(x_predict)
                if predictions.shape[1] == 2:
                    predictions = predictions[:, 1]
            else:
                predictions = np.asarray(fitted_model.predict(x_predict), dtype=np.float32)
            systemutils.loggerinfo(
                logger,
                f"[{self.__class__.__name__}] [predict] TabPFN Model prediction takes {(time.perf_counter() - st):.6f} seconds",
            )
            return predictions

        # --- pred_contribs path ---
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
            labels=self._encode_labels(),
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

        # Compute bias in the same logit space that produced phi.
        # The Shapley imputer uses ModelInferenceCallable(logits=True), which
        # calls predict_logits (native raw logits) for TabPFNClassifier, or
        # _proba_to_logits (manual conversion) for TabPFNManyClassifier.
        # We must use the same path here so bias = f(x) - sum(phi) holds.
        if self.is_classification:
            logit_fn = self._get_model_inference(fitted_model, logits=True)
            if self.is_many_classification:
                systemutils.loggerwarning(
                    logger,
                    f"[{self.__class__.__name__}] Shapley bias uses probability-to-logit "
                    f"conversion for ManyClassifier (>10 classes). "
                    f"Minor numerical approximation may occur.",
                )
            predictions_logit = logit_fn(x_predict)
            # MarginalImputer extracts [:, 1] for binary — match that here
            if predictions_logit.ndim == 2 and predictions_logit.shape[1] == 2:
                predictions_logit = predictions_logit[:, 1]
        else:
            predictions_logit = np.asarray(fitted_model.predict(x_predict), dtype=np.float32)
        bias = predictions_logit - phi.sum(axis=-1)
        bias = np.expand_dims(bias, axis=-1)
        pred_shap = np.concatenate((phi, bias), axis=-1).astype(np.float32)
        if pred_shap.ndim == 3:
            pred_shap = pred_shap.reshape(pred_shap.shape[0], -1)
        return pred_shap

    def _prepare_x(self, x: dt.Frame, encoder: OrdinalEncoder, y: Optional[np.ndarray] = None,
                   logger: Optional[logging.Logger] = None,
                   numeric_col_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        sample_indices = np.arange(x.shape[0])
        n = x.shape[0]

        if y is not None and n > self.TRAIN_SIZE_LIMITS:
            discard_pct = 100.0 * (1.0 - self.TRAIN_SIZE_LIMITS / n)
            if discard_pct >= 50.0:
                systemutils.loggerwarning(
                    logger,
                    f"[{self.__class__.__name__}] Downsampling from {n} to {self.TRAIN_SIZE_LIMITS} rows "
                    f"({discard_pct:.1f}% of data discarded)."
                )
            if self.is_classification:
                try:
                    sample_indices, _ = train_test_split(
                        np.arange(n),
                        train_size=self.TRAIN_SIZE_LIMITS,
                        stratify=y,
                        random_state=self.random_state,
                    )
                except ValueError:
                    # Stratified split fails when a class has < 2 samples;
                    # fall back to uniform random sampling.
                    rng = np.random.RandomState(self.random_state)
                    sample_indices = rng.choice(n, size=self.TRAIN_SIZE_LIMITS, replace=False)
            else:
                rng = np.random.RandomState(self.random_state)
                sample_indices = rng.choice(n, size=self.TRAIN_SIZE_LIMITS, replace=False)
            # Subsample the frame before numeric conversion to avoid
            # OrdinalEncoder processing rows that will be discarded.
            x = x[sample_indices.tolist(), :]

        x_numpy = _to_numeric(x, encoder, numeric_col_mask=numeric_col_mask)
        return np.asarray(x_numpy, dtype=np.float32), sample_indices

    def _get_tabpfn_model(self, n_jobs: int, device: str, enc_labels: Optional[Sequence[int]], logger: Optional[logging.Logger] = None):
        n_estimators = self.params.get("n_estimators", 8)
        softmax_temperature = self.params.get("softmax_temperature", 0.9)
        balance_probabilities = self.params.get("balance_probabilities", False)
        average_before_softmax = self.params.get("average_before_softmax", False)
        tune_boundary_threshold = self.params.get("tune_boundary_threshold", False)
        calibrate_softmax_temperature = self.params.get("calibrate_softmax_temperature", False)
        n_estimators_redundancy = self.params.get("n_estimators_redundancy", 4)
        finetune_epochs = self.params.get("finetune_epochs", 30)
        finetune_learning_rate = self.params.get("finetune_learning_rate", 1e-5)

        systemutils.loggerinfo(
            logger,
            f"parameters: n_estimators = {n_estimators}, softmax_temperature = {softmax_temperature}, balance_probabilities = {balance_probabilities},"
            f" average_before_softmax = {average_before_softmax}, tune_boundary_threshold = {tune_boundary_threshold},"
            f" calibrate_softmax_temperature = {calibrate_softmax_temperature}, n_estimators_redundancy = {n_estimators_redundancy},"
            f" finetune_epochs = {finetune_epochs}, finetune_learning_rate = {finetune_learning_rate}"
        )
        tabpfn_model = self._build_tabpfn_model(
            seed=self.random_state,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            device=device,
            softmax_temperature=softmax_temperature,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            tune_boundary_threshold=tune_boundary_threshold,
            calibrate_softmax_temperature=calibrate_softmax_temperature,
            finetune_epochs=finetune_epochs,
            finetune_learning_rate=finetune_learning_rate,
            logger=logger,
            build_classifier=self.is_classification or self.is_many_classification,
            build_regressor=not self.is_classification and not self.is_many_classification,
        )

        if self.is_many_classification:
            return TabPFNManyClassifier(
                estimator=tabpfn_model,
                n_estimators=n_estimators,
                n_estimators_redundancy=n_estimators_redundancy,
                random_state=self.random_state,
                labels=[] if enc_labels is None else enc_labels,
                logger=logger,
            )
        return tabpfn_model

    def _encode_labels(self) -> Optional[Sequence[int]]:
        if not hasattr(self, '_cached_enc_labels'):
            if self.is_classification:
                assert self.labels is not None
                enc_labels = LabelEncoder().fit_transform(self.labels)
                self._cached_enc_labels = np.sort(np.unique(enc_labels))
            else:
                self._cached_enc_labels = self.labels
        return self._cached_enc_labels

    def _get_model_inference(self, model, logits: bool = False) -> InferenceFunction:
        return ModelInferenceCallable(
            model=model,
            is_classification=self.is_classification,
            logits=logits,
        )

    @staticmethod
    def _build_tabpfn_model(
        seed: int,
        n_jobs: int,
        n_estimators: int,
        device: str,
        softmax_temperature: float = 0.5,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        tune_boundary_threshold: bool = False,
        calibrate_softmax_temperature: bool = False,
        finetune_epochs: int = 30,
        finetune_learning_rate: float = 1e-5,
        logger: Optional[logging.Logger] = None,
        build_classifier: bool = True,
        build_regressor: bool = True,
    ):
        """
        Instantiate FinetunedTabPFNClassifier/Regressor that fine-tunes V2 weights
        on the downstream dataset during fit().
        """
        from tabpfn.finetuning import FinetunedTabPFNClassifier, FinetunedTabPFNRegressor

        clf_ckpt, reg_ckpt = TabPFNModel._ensure_weights_cached(
            build_classifier=build_classifier, build_regressor=build_regressor,
        )

        if build_classifier:
            systemutils.loggerinfo(
                logger,
                f"Instantiating Finetuned TabPFN Classifier. ckpt={clf_ckpt}, "
                f"epochs={finetune_epochs}, lr={finetune_learning_rate}",
            )
            return FinetunedTabPFNClassifier(
                device=device,
                epochs=finetune_epochs,
                learning_rate=finetune_learning_rate,
                random_state=seed,
                n_estimators_final_inference=n_estimators,
                extra_classifier_kwargs={
                    "model_path": str(clf_ckpt),
                    "n_preprocessing_jobs": n_jobs,
                    "softmax_temperature": softmax_temperature,
                    "balance_probabilities": balance_probabilities,
                    "average_before_softmax": average_before_softmax,
                    "tuning_config": {
                        "tune_decision_thresholds": tune_boundary_threshold,
                        "calibrate_temperature": calibrate_softmax_temperature,
                    },
                },
            )

        systemutils.loggerinfo(
            logger,
            f"Instantiating Finetuned TabPFN Regressor. ckpt={reg_ckpt}, "
            f"epochs={finetune_epochs}, lr={finetune_learning_rate}",
        )
        return FinetunedTabPFNRegressor(
            device=device,
            epochs=finetune_epochs,
            learning_rate=finetune_learning_rate,
            random_state=seed,
            n_estimators_final_inference=n_estimators,
            extra_regressor_kwargs={
                "model_path": str(reg_ckpt),
                "n_preprocessing_jobs": n_jobs,
                "average_before_softmax": average_before_softmax,
            },
        )

    @staticmethod
    def _ensure_weights_cached(
        build_classifier: bool = True,
        build_regressor: bool = True,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Download needed checkpoints into a deterministic cache location.

        Only downloads checkpoints that are actually required, avoiding ~500 MB
        of unnecessary transfer on cold cache.
        """
        cache_dir = _get_cache_dir()
        systemutils.makedirs(cache_dir, exist_ok=True)

        clf_path = cache_dir / os.path.basename(TABPFN_CLASSIFIER_CKPT_URL)
        reg_path = cache_dir / os.path.basename(TABPFN_REGRESSOR_CKPT_URL)

        for url, final_path, expected_sha, needed in [
            (TABPFN_CLASSIFIER_CKPT_URL, clf_path, TABPFN_CLASSIFIER_CKPT_SHA256, build_classifier),
            (TABPFN_REGRESSOR_CKPT_URL, reg_path, TABPFN_REGRESSOR_CKPT_SHA256, build_regressor),
        ]:
            if not needed:
                continue
            if not final_path.exists():
                # Atomic download: write to PID-stamped temp file, verify, then rename.
                # Prevents concurrent workers from reading partially-written or corrupted files.
                tmp_path = final_path.with_suffix(f".tmp.{os.getpid()}")
                try:
                    download(url, dest=str(tmp_path))
                    _verify_checkpoint(tmp_path, expected_sha)
                    os.replace(str(tmp_path), str(final_path))
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise

        return (
            str(clf_path) if build_classifier else None,
            str(reg_path) if build_regressor else None,
        )

    @staticmethod
    def _get_device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

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
    def _prepare_y(
        y: np.ndarray,
        is_classification: bool,
        label_encoder: Optional[LabelEncoder] = None,
    ) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
        transformed_y = y.copy()
        if transformed_y.dtype.kind in {"U", "S", "O"} or is_classification:
            if label_encoder is None:
                label_encoder = LabelEncoder()
                transformed_y = label_encoder.fit_transform(transformed_y.astype(str))
            else:
                y_str = transformed_y.astype(str)
                unseen = set(y_str) - set(label_encoder.classes_)
                if unseen:
                    # Eval set has classes not seen during training (rare with
                    # stratified splitting).  Fall back to a fresh encoder so we
                    # don't crash; SAGE values may be slightly less accurate.
                    label_encoder = LabelEncoder()
                    transformed_y = label_encoder.fit_transform(y_str)
                else:
                    transformed_y = label_encoder.transform(y_str)
            transformed_y = transformed_y.astype(np.int64, copy=False)
        return transformed_y, label_encoder


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
    _LOGIT_EPS = 1e-7

    def __init__(self, model, is_classification: bool, logits: bool = False):
        """
        Args:
            model: The fitted TabPFN model (Classifier/Regressor/ManyClassifier)
            is_classification: Whether this is a classification task
            logits: Whether to return logits instead of probabilities
                    (classification only). Uses native predict_logit when
                    available (TabPFNClassifier), falls back to manual
                    conversion for TabPFNManyClassifier (>10 classes).
        """
        self.model = model
        self.is_classification = is_classification
        self.logits = logits

    @staticmethod
    def _proba_to_logits(proba: np.ndarray, eps: float = _LOGIT_EPS) -> np.ndarray:
        """Convert probabilities to logit space.

        - Binary (2 classes): uses log-odds log(p/(1-p)), the sigmoid inverse.
        - Multiclass (>2 classes): uses log(p), the log-softmax inverse.
        """
        proba = np.clip(proba, eps, 1.0 - eps)
        if proba.ndim == 2 and proba.shape[1] > 2:
            return np.log(proba)
        return np.log(proba / (1.0 - proba))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Invoke model inference on input data."""
        if self.is_classification:
            if self.logits and hasattr(self.model, 'predict_logits'):
                # Native TabPFN classifier supports direct logit output.
                # Returns shape (n_samples, n_classes) for both binary and multiclass.
                return self.model.predict_logits(x)
            proba = self.model.predict_proba(x)
            if self.logits:
                # Fallback for models without native logit support
                # (e.g., TabPFNManyClassifier which exceeds TabPFN's 10-class limit)
                return self._proba_to_logits(proba)
            return proba
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

    x_clean = X
    if np.isnan(X).any():
        x_clean = np.nan_to_num(X, nan=0.0)
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=2048, n_init="auto")
    km.fit(x_clean)

    # snap each center to nearest actual row (medoid-like)
    labels = km.predict(x_clean)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(x_clean), k, replace=False)
    assigned = np.zeros(k, dtype=bool)
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        if cluster_mask.any():
            cluster_points = x_clean[cluster_mask]
            # Find closest point to center
            distances = np.linalg.norm(cluster_points - km.cluster_centers_[cluster_id], axis=1)
            idx[cluster_id] = np.where(cluster_mask)[0][distances.argmin()]
            assigned[cluster_id] = True

    # Replace empty-cluster slots with unique random samples.
    # Guard k < len(X) (line above) guarantees enough available indices.
    if not assigned.all():
        used = set(idx[assigned].tolist())
        available = np.array([i for i in range(len(X)) if i not in used])
        n_empty = int((~assigned).sum())
        idx[~assigned] = rng.choice(available, size=n_empty, replace=False)

    # Deduplicate: two clusters may snap to the same data point.
    unique_idx, first_pos = np.unique(idx, return_index=True)
    if len(unique_idx) < k:
        used = set(unique_idx.tolist())
        available = np.array([i for i in range(len(X)) if i not in used])
        n_dup = k - len(unique_idx)
        extra = rng.choice(available, size=n_dup, replace=False)
        dup_mask = np.ones(k, dtype=bool)
        dup_mask[first_pos] = False
        idx[dup_mask] = extra

    return X[idx].astype(np.float32)


def _fit_encoder(x: dt.Frame, encoder: OrdinalEncoder, numeric_col_mask: np.ndarray):
    """Fit OrdinalEncoder on non-numeric columns of the full frame.

    Must be called before subsampling so the encoder sees all categories.
    """
    non_numeric = np.where(~numeric_col_mask)[0].tolist()
    if not non_numeric:
        return
    x_cat = x[:, non_numeric].to_numpy().astype(np.object_)
    encoder.fit(x_cat)


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
