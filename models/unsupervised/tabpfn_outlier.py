"""
TabPFN-based outlier scoring transformer for Driverless AI.

License compliance note (Prior Labs License v1.2, See license text; ensure compliance with attribution requirements):
- This recipe depends on `tabpfn` / `tabpfn-extensions` and may download/use TabPFN weights.
- If you DISTRIBUTE or make available a product/service containing TabPFN source/weights (or derivative work),
    you must satisfy the license additional attribution requirement (Section 10), including prominently displaying:
    “Built with PriorLabs-TabPFN” in relevant UI/docs.
"""

import os
import pathlib
import random
import uuid
from typing import List, Optional, Tuple

import datatable as dt
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from h2oaicore import systemutils
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.systemutils_more import download
from h2oaicore.transformer_utils import CustomTransformer


TABPFN_CLASSIFIER_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
)
TABPFN_REGRESSOR_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-regressor.ckpt"
)


class TabPFNOutliersDetection(BaseEstimator):
    MAX_CLASSES = 10
    """
    Simplified implementation of https://github.com/PriorLabs/tabpfn-extensions/blob/a54bc14398a5155ae22a5c0ac2fb9327e88782a8/src/tabpfn_extensions/unsupervised/unsupervised.py#L63
    Outlier-only subset of TabPFNUnsupervisedModel.

    Usage:
      clf = TabPFNClassifier(...)
      reg = TabPFNRegressor(...)
      imp = TabPFNOutliersDetection()
      imp.fit(X_train)
      # negative logarithm of conditional probabilities
      X_outlier_scores = imp.outliers(x=X_nan)
    """

    @staticmethod
    def load_from_bytes(model_bytes) -> "TabPFNOutliersDetection":
        return systemutils.load_obj_bytes(model_bytes)

    def __init__(self, eps: float = 1e-10, seed: int = 0, num_classes: int = 0, logger=None):
        self._eps = eps
        self._seed = seed
        self._logger = logger
        self._num_classes = num_classes

        self.train_x_ = None
        self.ord_encoder_ = None

    def to_bytes(self):
        return systemutils.save_obj_to_bytes(self)

    def fit(self, X: dt.Frame, y=None):
        assert len(X.shape) == 2

        self.ord_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        # OPTIMIZATION: Use in-place conversion to avoid extra copy
        x_numpy = _to_numeric(X, self.ord_encoder_)
        # Directly assign as float32 without intermediate copy if already correct dtype
        self.train_x_ = np.asarray(x_numpy, dtype=np.float32)

        # Clean up intermediate data
        del x_numpy

        return self

    def outliers(
        self,
        x: dt.Frame,
        n_permutations: int = 10,
        fast_mode: bool = False,
        seed: int = 0,
        n_jobs: int = 1,
        n_estimators: int = 8,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Calculate the negative logarithm outlier scores for each sample in the input data.

        This is the preferred implementation for outlier detection, which calculates
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
            n_jobs: int, default=1
                Number of jobs to run in parallel.
            n_estimators: int, default=8
                Number of estimators in TabPFN ensemble model.
            balance_probabilities: bool, default=False
                Whether to balance the probabilities based on the class distribution
                in the training data.
            average_before_softmax: bool, default=False
                Whether to average the predictions of
                the estimators before applying the softmax function.
            chunk_size: int, default=1000
                Number of samples to process at once. Larger values may be faster but use more memory.
                Smaller values reduce memory usage for large prediction datasets.

        Returns:
            np.ndarray:
                Array of negative logarithm outlier scores (higher values indicate more likely outliers),
                shape (n_samples,)

        Raises:
            RuntimeError: If the model initialization fails
            ValueError: If the input data has incompatible dimensions
        """
        import gc
        import torch

        assert len(x.shape) == 2
        check_is_fitted(self, attributes=["train_x_", "ord_encoder_"])

        device = self._get_device()
        seed = self._seed if seed is None else seed
        x_numpy = _to_numeric(x, self.ord_encoder_)
        x_numpy = np.asarray(x_numpy, dtype=np.float32)

        n_samples = x.shape[0]
        n_features = x.shape[1]
        all_features = list(range(n_features))

        # Use fewer permutations in fast mode
        actual_n_permutations = 1 if fast_mode else n_permutations

        # OPTIMIZATION: Process in chunks to handle large prediction datasets
        # This allows unbounded dataset sizes with bounded memory usage
        n_chunks = (n_samples + chunk_size - 1) // chunk_size

        # Determine if chunking is needed
        use_chunking = n_samples > chunk_size

        if use_chunking:
            systemutils.loggerinfo(
                self._logger,
                f"Processing {n_samples} samples in {n_chunks} chunks of size {chunk_size}..."
            )

        all_chunk_scores = []

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
            x_chunk = x_numpy[start_idx:end_idx]

            if use_chunking:
                systemutils.loggerinfo(
                    self._logger,
                    f"Processing chunk {chunk_idx + 1}/{n_chunks} (samples {start_idx}:{end_idx})..."
                )

            # OPTIMIZATION: Use streaming accumulation for each chunk
            # This reduces memory from O(n_perm × chunk_size) to O(chunk_size)
            log_density_sum = np.zeros(x_chunk.shape[0], dtype=np.float32)

            systemutils.loggerinfo(self._logger, "Permutate features now...")
            for perm_idx, perm in enumerate(_efficient_random_permutation(all_features, actual_n_permutations, seed)):
                perm_density_log, perm_density = self.outliers_single_permutation_(
                    x=x_chunk,
                    feature_permutation=perm,
                    n_jobs=n_jobs,
                    n_estimators=n_estimators,
                    device=device,
                    balance_probabilities=balance_probabilities,
                    average_before_softmax=average_before_softmax,
                    seed=seed,
                )

                # Incremental averaging: clean and accumulate immediately
                perm_density_log = np.nan_to_num(perm_density_log, nan=0.0, posinf=1e30, neginf=1e-30)
                log_density_sum += perm_density_log

                # OPTIMIZATION: Explicit cleanup - free memory after each permutation
                del perm_density_log, perm_density

                # Force garbage collection periodically
                if perm_idx % max(1, actual_n_permutations // 4) == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            systemutils.loggerinfo(self._logger, "Permutate features finished!")

            # Compute negative mean for this chunk
            chunk_scores = (log_density_sum / actual_n_permutations) * -1
            all_chunk_scores.append(chunk_scores)

            # OPTIMIZATION: Clean up chunk data before next iteration
            del x_chunk, log_density_sum, chunk_scores
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all chunk results
        final_scores = np.concatenate(all_chunk_scores)

        if use_chunking:
            systemutils.loggerinfo(self._logger, f"Finished processing all {n_chunks} chunks.")

        return final_scores

    def density_(
        self,
        x_predict: np.ndarray,
        x_fit: np.ndarray,
        conditional_idx: Tuple[int, ...],
        column_idx: int,
        n_jobs: int,
        n_estimators: int,
        device: str,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
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
            n_jobs: Number of jobs to run in parallel.
            n_estimators: Number of estimators in TabPFN ensemble model.
            device: Device to use.
            balance_probabilities: Whether to balance the probabilities based on the class distribution in the training data.
            average_before_softmax: Whether to average the predictions of the estimators before applying the softmax function.
            seed: Random seed, default=0

        Returns:
            tuple containing:
                - The fitted model (classifier or regressor)
                - The filtered features used for prediction
                - The target feature values to predict
                - Whether it is classification
        """
        if len(conditional_idx) > 0:
            # If not the first feature, use all previous features
            x_fit, y_fit = x_fit[:, conditional_idx], x_fit[:, column_idx]
            x_fit = x_fit.reshape(x_fit.shape[0], -1)

            x_predict, y_predict = x_predict[:, conditional_idx], x_predict[:, column_idx]
            x_predict = x_predict.reshape(x_predict.shape[0], -1)
        else:
            seed = self._seed if seed is None else seed
            rng = np.random.RandomState(seed)
            # If the first feature, use a zero feature as input
            # Because of preprocessing, we can't use a zero feature, so we use a random feature
            x_fit, y_fit = (
                rng.random(x_fit[:, 0:1].shape).astype(np.float32),
                x_fit[:, 0],
            )
            x_predict, y_predict = rng.random(x_predict[:, 0:1].shape).astype(np.float32), x_predict[:, 0]

        # Handle potential nan values in y_fit
        if np.isnan(y_fit).any():
            y_fit = np.nan_to_num(y_fit, nan=0.0)

        tabpfn_clf, tabpfn_reg = self._build_tabpfn_models(
            seed=seed,
            n_jobs=n_jobs,
            n_estimators=n_estimators,
            device=device,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
        )
        model = tabpfn_reg
        is_classification = self._is_classifier(y_fit)
        if is_classification:
            y_fit = y_fit.astype(np.int64)
            y_predict = y_predict.astype(np.int64)
            model = tabpfn_clf
        model.fit(x_fit, y_fit)

        return model, x_predict, y_predict, is_classification

    def _is_classifier(self, targets: np.ndarray) -> bool:
        # classifier if low cardinality <= TABPFN maximum classes
        return np.unique(targets).size <= self.MAX_CLASSES

    def outliers_single_permutation_(
        self,
        x: np.ndarray,
        feature_permutation: Tuple[int, ...],
        n_jobs: int,
        n_estimators: int,
        device: str,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        seed: Optional[int] = None,
    ):
        import gc
        import torch

        # Start with a log probability of 0 (log(1) = 0)
        log_p = np.zeros(x[:, 0].shape, dtype=np.float32)
        seed = self._seed if seed is None else seed

        for i, column_idx in enumerate(feature_permutation):
            model, x_predict, y_predict, is_classification = self.density_(
                x_predict=x,
                x_fit=self.train_x_,
                conditional_idx=feature_permutation[:i],
                column_idx=column_idx,
                n_jobs=n_jobs,
                n_estimators=n_estimators,
                device=device,
                balance_probabilities=balance_probabilities,
                average_before_softmax=average_before_softmax,
                seed=seed + 1000 * i + int(column_idx),
            )
            if is_classification:
                # Get predictions and convert to torch tensor
                pred_np = model.predict_proba(x_predict)

                # Convert y_predict to indices for indexing the probabilities
                y_indices = y_predict.astype(np.int64)

                # Check indices are in bounds
                valid_indices = (y_indices >= 0) & (y_indices < pred_np.shape[1])
                # Get default probability filled with a reasonable value
                # Default small probability
                pred = np.full(x_predict.shape[0], 0.1, dtype=np.float32)
                rows = np.arange(x_predict.shape[0])
                # Only index with valid indices
                pred[valid_indices] = pred_np[rows[valid_indices], y_indices[valid_indices]]

                # OPTIMIZATION: Clean up intermediate arrays
                del pred_np, y_indices, valid_indices, rows
            else:
                pred = model.predict(x_predict, output_type="full")

                # Get logits tensor properly
                logits = pred["logits"]
                if hasattr(logits, "detach"):
                    logits_tensor = logits.detach()
                else:
                    logits_tensor = torch.as_tensor(logits)
                y_tensor = torch.tensor(y_predict).to(logits_tensor.device)

                pred_t = pred["criterion"].pdf(logits_tensor, y_tensor)
                pred = pred_t.detach().cpu().numpy().astype(np.float32)

                # OPTIMIZATION: Clean up tensors and intermediate results
                del logits, logits_tensor, y_tensor, pred_t

            # Handle zero or negative probabilities (avoid log(0))
            pred = np.clip(pred, self._eps, None)

            # Convert probabilities to log probabilities
            log_pred = np.log(pred)

            # Add log probabilities instead of multiplying probabilities
            log_p = log_p + log_pred

            # OPTIMIZATION: Explicit cleanup - free memory after each feature
            del model, x_predict, y_predict, pred, log_pred

            # Periodic garbage collection every 5 features to reduce memory pressure
            if i > 0 and i % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # OPTIMIZATION: Return exp(log_p) without keeping extra reference
        # Note: The caller only uses the first return value (log_p),
        # so we minimize memory by computing exp on-the-fly
        exp_log_p = np.exp(log_p)
        return log_p, exp_log_p

    @staticmethod
    def _build_tabpfn_models(
        seed: int,
        n_jobs: int,
        n_estimators: int,
        device: str,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
    ):
        """
        Instantiate TabPFNClassifier/Regressor in a way compatible with your environment.
        You may need to adjust constructor args to point to ckpt paths depending on TabPFN version.
        """
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        clf_ckpt, reg_ckpt = TabPFNOutliersDetection._ensure_weights_cached()

        systemutils.loggerinfo(None, f"Instantiating TabPFN Classifier and Regressor. {clf_ckpt} and {reg_ckpt}")
        tabpfn_clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2,
            device=device,
            model_path=clf_ckpt,
            random_state=seed,
            n_preprocessing_jobs=n_jobs,
            n_estimators=n_estimators,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            fit_mode="fit_with_cache", # Faster inference for GPU.
        )
        tabpfn_reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2,
            device=device,
            model_path=reg_ckpt,
            random_state=seed,
            n_preprocessing_jobs=n_jobs,
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            fit_mode="fit_with_cache", # Faster inference for GPU.
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


class TabPFNOutlierScoreTransformer(CustomTransformer):
    r"""
        TabPFN-based outlier score transformer for Driverless AI.

        CAUTION: TabPFN pretrained model has fitting size limitation, the upperbound to be max_fit_rows < 10000,
        > 10000 is technically feasible but performance can be jeopardized.
        GPU inference is highly recommend and this transformer works best to small dataset < 10000
        reference: https://github.com/PriorLabs/TabPFN

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
        2) prominently display: **“Built with PriorLabs-TabPFN”** on relevant UI/docs pages.
        (Internal benchmarking/testing without external communication does not qualify as distribution per the license.)

        Notes for DAI users:
        - This transformer is computationally expensive; keep `max_cols` and `n_permutations` conservative.
        - Output is numeric and can be used directly as an engineered feature for downstream models.
        """
    _unsupervised = True
    _numeric_output = True
    _is_reproducible = True
    _parallel_task = False
    _can_use_gpu = True
    _must_use_gpu = True
    _can_use_multi_gpu = False
    _force_no_fork_isolation = False
    _get_gpu_lock = True
    _get_gpu_lock_vis = True
    _mojo = False
    _display_name = "TabPFN (Unsupervised) Outliers Scoring Transformer"
    _testing_can_skip_failure = False
    _allow_transform_to_modify_output_feature_names = True
    _modules_needed_by_name = ["tabpfn==6.2.0"]

    TRAIN_SIZE_LIMITS = 1000
    """
    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0,
                num_classes=None, **kwargs):
        return accuracy > 8 and interpretability < 2 and train_shape[0] < 10000 and n_gpus > 0

    @staticmethod
    def enabled_setting():
        return "auto"
    """
    @staticmethod
    def do_acceptance_test():
        # Very slow, manually set be `True` for testing purpose
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="numcat", min_cols=2, max_cols=500, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return dict(
            n_permutations=[3, 5, 7],
            n_estimators=[8, 10, 12],
            balance_probabilities=[False, True],
            average_before_softmax=[False, True],
            eps=[1e-10],
            chunk_size=[500, 1000, 2000, 5000],
        )

    @property
    def display_name(self):
        return (f"TabPFNOutlierScore(p={self.n_permutations},"
                f"n_estimators={self.n_estimators},chunk_size={self.chunk_size},"
                f"balance_probabilities={self.balance_probabilities},"
                f"average_before_softmax={self.average_before_softmax})")

    def __init__(
        self,
        n_permutations: int = 5,
        n_estimators: int = 8,
        max_fit_rows: int = 10000,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        eps: float = 1e-12,
        chunk_size: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_permutations = n_permutations
        self.max_fit_rows = max_fit_rows
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.n_estimators = n_estimators
        self.eps = eps
        self.chunk_size = chunk_size
        self.uid = str(uuid.uuid4())
        self.seed = systemutils.config.seed

        # learned state
        self.raw_model_bytes = None
        self.detector_ = None

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

        self._prepare_env(limit=X.shape[0])
        x, sample_indices = self._prepare_x(X)

        self.detector_ = TabPFNOutliersDetection(
            eps=self.eps,
            seed=self.seed,
            num_classes=0 if self.labels is None else len(self.labels),
            logger=logger,
        )
        systemutils.loggerinfo(logger, "Fitting outlier detector...")
        self.detector_.fit(x[sample_indices, :])

        systemutils.loggerinfo(logger, "Fitted outlier detector. Scoring...")
        scores = self._scores(x, logger=logger, **kwargs)

        systemutils.loggerinfo(logger, "Transforming final output...")
        final_output = self._transform(scores=scores, sample_indices=sample_indices, full=x.shape[0])

        systemutils.loggerinfo(logger, "Saving model...")
        self._save_model()

        return final_output

    def transform(self, X: dt.Frame, **kwargs) -> np.ndarray:
        assert len(X.shape) == 2
        check_is_fitted(self, ["raw_model_bytes"])
        self.detector_ = TabPFNOutliersDetection.load_from_bytes(self.raw_model_bytes)
        assert self.detector_ is not None

        self._prepare_env()
        x, _ = self._prepare_x(X)
        scores = self._scores(x, logger=None, **kwargs)
        return self._transform(scores=scores, sample_indices=np.arange(x.shape[0]), full=x.shape[0])

    def _transform(self, scores: np.ndarray, full: int, sample_indices: np.ndarray) -> np.ndarray:
        self._output_feature_names = ["OutlierScore"]
        self._feature_desc = ["Negative Logarithm of row conditional probabilities"]

        final_output = scores.reshape(-1, 1)
        if full > final_output.shape[0]:
            finals = np.full((full, 1), fill_value=0.0, dtype=np.float32)
            finals[sample_indices] = final_output
            return finals
        return final_output

    def _scores(self, x: dt.Frame, logger = None, **kwargs) -> np.ndarray:
        scores = self.detector_.outliers(
            x=x,
            n_permutations=self.n_permutations,
            seed=self.seed,
            n_jobs=self._get_n_jobs(logger, **kwargs),
            n_estimators=self.n_estimators,
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            #chunk_size=self.chunk_size,
            chunk_size=x.shape[0], # TODO: remove chunk_size
        )
        assert scores.shape[0] == x.shape[0]
        return scores

    def _save_model(self):
        self.raw_model_bytes = self.detector_.to_bytes()

    def _prepare_env(self, limit: int = -1):
        import torch

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        use_gpu = torch.cuda.is_available()
        if use_gpu:
            torch.cuda.manual_seed_all(self.seed)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
            os.putenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "true"
        os.putenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")

        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
        os.putenv("TABPFN_DISABLE_TELEMETRY", "1")
        os.environ["TABPFN_MODEL_CACHE_DIR"] = str(_get_cache_dir())
        os.putenv("TABPFN_MODEL_CACHE_DIR", str(_get_cache_dir()))
        if not use_gpu and limit > self.TRAIN_SIZE_LIMITS:
            os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
            os.putenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    def _prepare_x(self, x: dt.Frame) -> Tuple[dt.Frame, np.ndarray]:
        sample_indices = np.arange(x.shape[0])
        n = x.shape[0]

        if n > self.max_fit_rows:
            rng = np.random.RandomState(self.seed)
            sample_indices = rng.choice(n, self.max_fit_rows, replace=False)

        return x, sample_indices

    @staticmethod
    def _get_n_jobs(logger, **kwargs) -> int:
        try:
            if systemutils.config.fixed_num_folds <= 0:
                n_jobs = max(1, int(int(systemutils.max_threads() / min(systemutils.config.num_folds, kwargs['max_workers']))))
            else:
                n_jobs = max(1, int(
                    int(systemutils.max_threads() / min(systemutils.config.fixed_num_folds, systemutils.config.num_folds, kwargs['max_workers']))))
        except KeyError:
            systemutils.loggerinfo(logger, "Arima No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        return n_jobs


class TabPFNOutlierScoreModel(CustomUnsupervisedModel):
    _included_pretransformers = ['OrigFreqPreTransformer']  # frequency-encode categoricals, keep numerics as is
    _included_transformers = ["TabPFNOutlierScoreTransformer"]
    _included_scorers = ['UnsupervisedScorer']


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
    random.seed(seed)
    # Create a copy of the list to avoid modifying the original
    permutation = list(indices)

    # Shuffle the list in-place using Fisher-Yates algorithm
    for i in range(len(indices) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i)
        # Swap elements at i and j
        permutation[i], permutation[j] = permutation[j], permutation[i]

    return tuple(permutation)


def _to_numeric(x: dt.Frame, ord_encoder: OrdinalEncoder) -> np.ndarray:
    assert len(x.shape) == 2

    if x.shape[0] == 0:
        return x.to_numpy().astype(np.float32)

    numeric_col_mask = _numeric_column_mask(x)
    if np.all(numeric_col_mask):
        return x.to_numpy().astype(np.float32)

    numeric_col_indices = np.where(numeric_col_mask)[0].tolist()
    non_numeric_col_indices = np.where(~numeric_col_mask)[0].tolist()

    # OPTIMIZATION: Pre-allocate with correct dtype to avoid copies
    numeric_array = np.empty((x.nrows, x.ncols), dtype=np.float32)

    if numeric_col_indices:
        # OPTIMIZATION: Direct assignment without intermediate conversion
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

    # OPTIMIZATION: Use in-place assignment with proper casting
    numeric_array[:, non_numeric_col_indices] = np.asarray(numeric_transformed, dtype=np.float32)

    # Clean up intermediate data
    del x_cat, numeric_transformed

    return numeric_array


def _numeric_column_mask(x: dt.Frame) -> np.ndarray:
    mask = np.zeros(x.shape[1], dtype=bool)
    for i in range(x.shape[1]):
        col = x[:, i].to_numpy()
        try:
            col.astype(np.float32)  # succeeds for numeric-like strings too
            mask[i] = True
        except (ValueError, TypeError):
            mask[i] = False
    return mask


def _get_cache_dir() -> pathlib.Path:
    return pathlib.Path(systemutils.temporary_files_abspath) / "tabpfn_cache"
