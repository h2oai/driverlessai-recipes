"""
TabPFN-based embedding transformer for Driverless AI.

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
from typing import Optional
from typing import Tuple

import datatable as dt
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from h2oaicore import systemutils
from h2oaicore import transformers
from h2oaicore.systemutils_more import download
from h2oaicore.transformer_utils import CustomTransformer


TABPFN_CLASSIFIER_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
)
TABPFN_REGRESSOR_CKPT_URL = (
    "https://s3.amazonaws.com/artifacts.h2o.ai/releases/ai/h2o/pretrained/tabpfn/tabpfn-v2-regressor.ckpt"
)


class TabPFNEmbeddingTransformer(CustomTransformer):
    r"""
        TabPFN-based embedding transformer for Driverless AI.

        CAUTION: TabPFN pretrained model has fitting size limitation, the upperbound to be max_fit_rows < 10000,
        > 10000 is technically feasible but performance can be jeopardized.
        GPU inference is highly recommend and this transformer works best to small dataset < 10000
        reference: https://github.com/PriorLabs/TabPFN

        **What it does**
        - Fits a supervised TabPFNRegressor/TabPFNClassifier on the selected feature columns.
        - Extracted embeddings from the fitted model.

        **References**
        - Upstream TabPFN utilities:
          https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/embedding/tabpfn_embedding.py
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
        - Output is numeric and can be used directly as an engineered feature for downstream models.
    """
    _numeric_output = True
    _is_reproducible = True
    _parallel_task = False
    _can_use_gpu = True
    _must_use_gpu = True
    _can_use_multi_gpu = False
    _force_no_fork_isolation = False
    _mojo = False
    _display_name = "TabPFN Embedding Transformer"
    _testing_can_skip_failure = False
    _unsupervised = False  # uses target
    _get_gpu_lock = True
    _get_gpu_lock_vis = True
    _uses_target = True
    _modules_needed_by_name = ["tabpfn==6.2.0"]

    TRAIN_SIZE_LIMITS = 10000
    TRAIN_SIZE_OVERLOAD_RATE = 2
    MAX_CLASSES = 10
    MAX_FEATURES = 30

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0,
                num_classes=None, **kwargs):
        return (
            accuracy > 8
            and interpretability < 2
            and train_shape[0] < TabPFNEmbeddingTransformer.TRAIN_SIZE_OVERLOAD_RATE * TabPFNEmbeddingTransformer.TRAIN_SIZE_LIMITS
            and train_shape[1] <= TabPFNEmbeddingTransformer.MAX_FEATURES
            and n_gpus > 0
        )

    @staticmethod
    def enabled_setting():
        return "auto"

    @staticmethod
    def do_acceptance_test():
        # Very slow, manually set be `False` to skip
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="numcat", min_cols=2, max_cols=TabPFNEmbeddingTransformer.MAX_FEATURES, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return dict(
            n_estimators=[8, 10, 12],
            balance_probabilities=[False, True],
            average_before_softmax=[False, True],
            tune_boundary_threshold=[False, True],
            calibrate_softmax_temperature=[False, True],
            max_dim=[40, 60, 80],
            pooling_type=["mean", "max"],
        )

    @property
    def display_name(self):
        return (f"TabPFNEmbedding(n_estimators={self.n_estimators},max_dim={self.max_dim},tune_boundary_threshold={self.tune_boundary_threshold},"
                f"pooling_type={self.pooling_type},balance_probabilities={self.balance_probabilities},"
                f"average_before_softmax={self.average_before_softmax},calibrate_softmax_temperature={self.calibrate_softmax_temperature})")

    @property
    def is_classification(self) -> bool:
        num_classes = len(self.labels or [])
        return num_classes > 1

    def __init__(
        self,
        n_estimators: int = 8,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        tune_boundary_threshold: bool = False,
        calibrate_softmax_temperature: bool = False,
        max_dim: int = 40,
        pooling_type: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.calibrate_softmax_temperature = calibrate_softmax_temperature
        self.tune_boundary_threshold = tune_boundary_threshold
        self.max_dim = max_dim
        self.pooling_type = pooling_type
        self.max_fit_rows = self.TRAIN_SIZE_LIMITS
        self.uid = str(uuid.uuid4())
        self.seed = systemutils.config.seed

        # learned state
        self.raw_model_bytes: Optional[bytes] = None
        self.ord_encoder_: Optional[OrdinalEncoder] = None
        self.tabpfn_model_ = None
        self.svd_ = None

    def fit_transform(self, X: dt.Frame, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        assert len(X.shape) == 2
        assert y is not None

        logger = None
        if self.context and self.context.experiment_id:
            logger = systemutils.make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        self._prepare_env(seed=self.seed)

        self.ord_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )
        x_numpy, sample_indices = self._prepare_x(x=X, encoder=self.ord_encoder_, y=y)
        x_sampled_numpy = x_numpy[sample_indices]

        device = self._get_device()
        self.tabpfn_model_ = self._build_tabpfn_models(
            seed=self.seed,
            n_estimators=self.n_estimators,
            n_jobs=self._get_n_jobs(logger, **kwargs),
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            tune_boundary_threshold=self.tune_boundary_threshold,
            calibrate_softmax_temperature=self.calibrate_softmax_temperature,
            is_classification=self.is_classification,
            device=device,
        )
        y_sampled = self._prepare_y(y[sample_indices], self.is_classification)

        systemutils.loggerinfo(logger, f"Fitting TabPFN {'Classifier' if self.is_classification else 'Regressor'}...")
        self.tabpfn_model_.fit(x_sampled_numpy, y_sampled)

        x_transformed = self._transform(x_numpy, data_source="test", training=True, use_gpu=device != "cpu")

        # Validate embeddings for NaN/inf
        if np.any(np.isnan(x_transformed)) or np.any(np.isinf(x_transformed)):
            systemutils.loggerwarning(logger, "WARNING: NaN or inf detected in embeddings, replacing with 0.0")
            x_transformed = np.nan_to_num(x_transformed, nan=0.0, posinf=0.0, neginf=0.0)

        self._save_state()
        systemutils.loggerinfo(logger, f"Finished fitting TabPFN {'Classifier' if self.is_classification else 'Regressor'}")

        return np.asarray(x_transformed, dtype=np.float32)

    def transform(self, X: dt.Frame, **kwargs) -> np.ndarray:
        assert len(X.shape) == 2
        check_is_fitted(self, ["raw_model_bytes"])

        self._restore_state()
        self._prepare_env(seed=self.seed)
        x_numpy, _ = self._prepare_x(x=X, encoder=self.ord_encoder_, y=None)
        x_transformed = self._transform(x_numpy, data_source="test", use_gpu=self._get_device() != "cpu")

        # Validate embeddings for NaN/inf
        if np.any(np.isnan(x_transformed)) or np.any(np.isinf(x_transformed)):
            systemutils.loggerwarning(None, "WARNING: NaN or inf detected in embeddings during transform, replacing with 0.0")
            x_transformed = np.nan_to_num(x_transformed, nan=0.0, posinf=0.0, neginf=0.0)

        return x_transformed

    def _transform(self, X: np.ndarray, data_source: str, training: bool = False, use_gpu: bool = False) -> np.ndarray:
        if len(X.shape) == 1:
            X = X[:, None]

        x_transformed = self.tabpfn_model_.get_embeddings(X, data_source=data_source)
        if len(x_transformed.shape) == 2:
            x_transformed = x_transformed[:, None, :]
        x_transformed = np.swapaxes(x_transformed, 0, 1)

        # Apply pooling: mean or max across ensemble estimators
        if self.pooling_type == "max":
            x_transformed = np.asarray(x_transformed.max(axis=1), dtype=np.float32)
        else:  # default to mean
            x_transformed = np.asarray(x_transformed.mean(axis=1), dtype=np.float32)

        if training:
            self._init_svd(x_transformed.shape[-1], use_gpu)
            final_output = self.svd_.fit_transform(x_transformed)
        else:
            final_output = self.svd_.transform(x_transformed)

        self._claim_memory()
        return final_output

    def _init_svd(self, num_features: int, use_gpu: bool):
        if systemutils.config.enable_h2o4gpu_truncatedsvd and use_gpu:
            self.svd_ = transformers.GPUTruncatedSVD(
                n_components=min(self.max_dim, num_features),
                algorithm=["power", "arpack"],
                tol=[1e-2, 0],
                n_iter=[30, 5],
                n_gpus=1,
                gpu_id=self._get_gpu_id(),
                random_state=self.seed,
                verbose=True if systemutils.config.debug_h2o4gpu_level > 0 else False,
            )
        else:
            self.svd_ = transformers.CPUTruncatedSVD(
                n_components=min(self.max_dim, num_features),
                algorithm="randomized",
                n_iter=5,
                tol=0.05,
                random_state=self.seed,
            )

    def _save_state(self):
        self.raw_model_bytes = systemutils.save_obj_to_bytes({
            "ord_encoder": self.ord_encoder_,
            "tabpfn_model_": self.tabpfn_model_,
            "svd_": self.svd_,
        })
        self._reset_state()

    def _reset_state(self):
        self.ord_encoder_ = None
        self.tabpfn_model_ = None
        self.svd_ = None

    def _restore_state(self):
        assert self.raw_model_bytes is not None

        if self.ord_encoder_ is None or self.tabpfn_model_ is None:
            model_state = systemutils.load_obj_bytes(self.raw_model_bytes)
            self.ord_encoder_ = model_state.get("ord_encoder", None)
            self.tabpfn_model_ = model_state.get("tabpfn_model_", None)
            self.svd_ = model_state.get("svd_", None)

        assert self.tabpfn_model_ is not None
        assert self.ord_encoder_ is not None
        assert self.svd_ is not None

    def _prepare_x(
        self,
        x: dt.Frame,
        encoder: OrdinalEncoder,
        y: Optional[np.ndarray] = None,
    ) -> Tuple[dt.Frame, np.ndarray]:
        """
        Prepare X for fitting with optional stratified sampling for classification.

        Args:
            x: Input features
            encoder: Categorical features encoder
            y: Target values (optional, used for stratified sampling in classification)

        Returns:
            Tuple of (x, sample_indices)
        """
        sample_indices = np.arange(x.shape[0])
        n = x.shape[0]

        if n > self.max_fit_rows:
            # Use stratified sampling for classification if y is provided
            if y is not None and self.is_classification:
                sample_indices, _ = train_test_split(
                    np.arange(n),
                    train_size=self.max_fit_rows,
                    stratify=y,
                    random_state=self.seed
                )
            else:
                # Uniform random sampling for regression or when y is not provided
                rng = np.random.RandomState(self.seed)
                sample_indices = rng.choice(n, self.max_fit_rows, replace=False)

        x_numpy = _to_numeric(x, encoder)
        return np.asarray(x_numpy, dtype=np.float32), sample_indices

    @staticmethod
    def _get_gpu_id() -> int:
        return (systemutils.get_gpu_id() + os.getpid() % systemutils.ngpus_vis) % systemutils.ngpus_vis_real

    @staticmethod
    def _get_n_jobs(logger, **kwargs) -> int:
        try:
            if systemutils.config.fixed_num_folds <= 0:
                n_jobs = max(1, int(int(
                    systemutils.max_threads() / min(systemutils.config.num_folds, kwargs['max_workers']))))
            else:
                n_jobs = max(1, int(
                    int(systemutils.max_threads() / min(systemutils.config.fixed_num_folds,
                                                        systemutils.config.num_folds, kwargs['max_workers']))))
        except KeyError:
            systemutils.loggerwarning(logger, "No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        return n_jobs

    @staticmethod
    def _build_tabpfn_models(
            seed: int,
            n_jobs: int,
            n_estimators: int,
            balance_probabilities: bool = False,
            average_before_softmax: bool = False,
            tune_boundary_threshold: bool = False,
            calibrate_softmax_temperature: bool = False,
            is_classification: bool = False,
            device: str = "cpu",
    ):
        from tabpfn.constants import ModelVersion

        clf_ckpt, reg_ckpt = TabPFNEmbeddingTransformer._ensure_weights_cached()
        if is_classification:
            from tabpfn import TabPFNClassifier

            systemutils.loggerinfo(None, f"Instantiating TabPFN Classifier from {clf_ckpt}")
            return TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                device=device,
                model_path=clf_ckpt,
                random_state=seed,
                n_preprocessing_jobs=n_jobs,
                n_estimators=n_estimators,
                balance_probabilities=balance_probabilities,
                average_before_softmax=average_before_softmax,
                tuning_config={"tune_decision_thresholds": tune_boundary_threshold,
                               "calibrate_temperature": calibrate_softmax_temperature},
            )
        else:
            from tabpfn import TabPFNRegressor

            systemutils.loggerinfo(None, f"Instantiating TabPFN Regressor from {reg_ckpt}")
            return TabPFNRegressor.create_default_for_version(
                ModelVersion.V2,
                device=device,
                model_path=reg_ckpt,
                random_state=seed,
                n_preprocessing_jobs=n_jobs,
                n_estimators=n_estimators,
                average_before_softmax=average_before_softmax,
            )

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
    def _claim_memory():
        import gc
        import torch

        if torch.cuda.is_available():
            try:
                allocated_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if allocated_memory > 0.8:
                    torch.cuda.empty_cache()
                    gc.collect()
                    return
            except (RuntimeError, AttributeError):
                # Fallback if memory stats unavailable
                torch.cuda.empty_cache()
        # CPU: collect garbage after all permutations for this chunk
        gc.collect()

    @staticmethod
    def _prepare_y(y: np.ndarray, is_classification: bool) -> np.ndarray:
        transformed_y = y.copy()
        if transformed_y.dtype.kind in {"U", "S", "O"} or is_classification:
            le = LabelEncoder()
            transformed_y = le.fit_transform(transformed_y.astype(str))
            transformed_y = transformed_y.astype(np.int64, copy=False)
        return transformed_y

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


def _get_cache_dir() -> pathlib.Path:
    return pathlib.Path(systemutils.temporary_files_abspath) / "tabpfn_cache"
