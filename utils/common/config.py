r"""
Miscellaneous utilities
"""

import logging
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from multiprocessing import Process

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
import torch
from anndata import AnnData
from pybedtools.helpers import set_bedtools_path
from sklearn.preprocessing import normalize

from .typehint import Array, RandomState, T


def normalize_edges(eidx: np.ndarray, ewt: np.ndarray, method: str = "keepvar") -> np.ndarray:
    r"""
    Normalize graph edge weights

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    method
        Normalization method, should be one of {"in", "out", "sym", "keepvar"}

    Returns
    -------
    enorm
        Normalized weight of edges (:math:`n_{edges}`)
    """
    if method not in ("in", "out", "sym", "keepvar"):
        raise ValueError("Unrecognized method!")
    enorm = ewt
    if method in ("in", "keepvar", "sym"):
        in_degrees = vertex_degrees(eidx, ewt, direction="in")
        in_normalizer = np.power(in_degrees[eidx[1]], -1 if method == "in" else -0.5)
        in_normalizer[~np.isfinite(in_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * in_normalizer
    if method in ("out", "sym"):
        out_degrees = vertex_degrees(eidx, ewt, direction="out")
        out_normalizer = np.power(out_degrees[eidx[0]], -1 if method == "out" else -0.5)
        out_normalizer[~np.isfinite(out_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * out_normalizer
    return enorm


def vertex_degrees(
    eidx: np.ndarray, ewt: np.ndarray, vnum: int | None = None, direction: str = "both"
) -> np.ndarray:
    r"""
    Compute vertex degrees

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    vnum
        Total number of vertices (determined by max edge index if not specified)
    direction
        Direction of vertex degree, should be one of {"in", "out", "both"}

    Returns
    -------
    degrees
        Vertex degrees
    """
    vnum = vnum or eidx.max() + 1
    adj = scipy.sparse.coo_matrix((ewt, (eidx[0], eidx[1])), shape=(vnum, vnum))
    if direction == "in":
        return adj.sum(axis=0).A1
    elif direction == "out":
        return adj.sum(axis=1).A1
    elif direction == "both":
        return adj.sum(axis=0).A1 + adj.sum(axis=1).A1 - adj.diagonal()
    raise ValueError("Unrecognized direction!")


def get_default_numpy_dtype() -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype

    Returns
    -------
    dtype
        Default numpy dtype
    """
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


# ------------------------------ Global containers ------------------------------

processes: Mapping[int, Mapping[int, Process]] = defaultdict(dict)  # id -> pid -> process


# -------------------------------- Meta classes ---------------------------------


class SingletonMeta(type):
    r"""
    Ensure singletons via a meta class
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


# --------------------------------- Log manager ---------------------------------


class _CriticalFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


class _NonCriticalFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < logging.WARNING


class LogManager(metaclass=SingletonMeta):
    r"""
    Manage loggers used in the package
    """

    def __init__(self) -> None:
        self._loggers = {}
        self._log_file = None
        self._console_log_level = logging.INFO
        self._file_log_level = logging.DEBUG
        self._file_fmt = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"
        self._console_fmt = "[%(levelname)s] %(name)s: %(message)s"
        self._date_fmt = "%Y-%m-%d %H:%M:%S"

    @property
    def log_file(self) -> str:
        r"""
        Configure log file
        """
        return self._log_file

    @property
    def file_log_level(self) -> int:
        r"""
        Configure logging level in the log file
        """
        return self._file_log_level

    @property
    def console_log_level(self) -> int:
        r"""
        Configure logging level printed in the console
        """
        return self._console_log_level

    def _create_file_handler(self) -> logging.FileHandler:
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.file_log_level)
        file_handler.setFormatter(logging.Formatter(fmt=self._file_fmt, datefmt=self._date_fmt))
        return file_handler

    def _create_console_handler(self, critical: bool) -> logging.StreamHandler:
        if critical:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.addFilter(_CriticalFilter())
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.addFilter(_NonCriticalFilter())
        console_handler.setLevel(self.console_log_level)
        console_handler.setFormatter(logging.Formatter(fmt=self._console_fmt))
        return console_handler

    def get_logger(self, name: str) -> logging.Logger:
        r"""
        Get a logger by name
        """
        if name in self._loggers:
            return self._loggers[name]
        new_logger = logging.getLogger(name)
        new_logger.setLevel(logging.DEBUG)  # lowest level
        new_logger.addHandler(self._create_console_handler(True))
        new_logger.addHandler(self._create_console_handler(False))
        if self.log_file:
            new_logger.addHandler(self._create_file_handler())
        self._loggers[name] = new_logger
        return new_logger

    @log_file.setter
    def log_file(self, file_name: os.PathLike) -> None:
        self._log_file = file_name
        for logger in self._loggers.values():
            for idx, handler in enumerate(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    logger.handlers[idx].close()
                    if self.log_file:
                        logger.handlers[idx] = self._create_file_handler()
                    else:
                        del logger.handlers[idx]
                    break
            else:
                if file_name:
                    logger.addHandler(self._create_file_handler())

    @file_log_level.setter
    def file_log_level(self, log_level: int) -> None:
        self._file_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(self.file_log_level)
                    break

    @console_log_level.setter
    def console_log_level(self, log_level: int) -> None:
        self._console_log_level = log_level
        for logger in self._loggers.values():
            for handler in logger.handlers:
                if type(handler) is logging.StreamHandler:  # pylint: disable=unidiomatic-typecheck
                    handler.setLevel(self.console_log_level)


log = LogManager()


def logged(obj: T) -> T:
    r"""
    Add logger as an attribute
    """
    obj.logger = log.get_logger(obj.__name__)
    return obj


# ---------------------------- Configuration Manager ----------------------------


@logged
class ConfigManager(metaclass=SingletonMeta):
    r"""
    Global configurations
    """

    def __init__(self) -> None:
        self.TMP_PREFIX = "GLUETMP"
        self.ANNDATA_KEY = "__scglue__"
        self.CPU_ONLY = False
        self.CUDNN_MODE = "repeatability"
        self.MASKED_GPUS = []
        self.ARRAY_SHUFFLE_NUM_WORKERS = 0
        self.GRAPH_SHUFFLE_NUM_WORKERS = 1
        self.FORCE_TERMINATE_WORKER_PATIENCE = 60
        self.DATALOADER_NUM_WORKERS = 0
        self.DATALOADER_FETCHES_PER_WORKER = 4
        self.DATALOADER_PIN_MEMORY = True
        self.CHECKPOINT_SAVE_INTERVAL = 10
        self.CHECKPOINT_SAVE_NUMBERS = 3
        self.PRINT_LOSS_INTERVAL = 10
        self.TENSORBOARD_FLUSH_SECS = 5
        self.ALLOW_TRAINING_INTERRUPTION = True
        self.BEDTOOLS_PATH = ""

    @property
    def TMP_PREFIX(self) -> str:
        r"""
        Prefix of temporary files and directories created.
        Default values is ``"GLUETMP"``.
        """
        return self._TMP_PREFIX

    @TMP_PREFIX.setter
    def TMP_PREFIX(self, tmp_prefix: str) -> None:
        self._TMP_PREFIX = tmp_prefix

    @property
    def ANNDATA_KEY(self) -> str:
        r"""
        Key in ``adata.uns`` for storing dataset configurations.
        Default value is ``"__scglue__"``
        """
        return self._ANNDATA_KEY

    @ANNDATA_KEY.setter
    def ANNDATA_KEY(self, anndata_key: str) -> None:
        self._ANNDATA_KEY = anndata_key

    @property
    def CPU_ONLY(self) -> bool:
        r"""
        Whether computation should use only CPUs.
        Default value is ``False``.
        """
        return self._CPU_ONLY

    @CPU_ONLY.setter
    def CPU_ONLY(self, cpu_only: bool) -> None:
        self._CPU_ONLY = cpu_only
        if self._CPU_ONLY and self._DATALOADER_NUM_WORKERS:
            self.logger.warning(
                "It is recommended to set `DATALOADER_NUM_WORKERS` to 0 "
                "when using CPU_ONLY mode. Otherwise, deadlocks may happen "
                "occationally."
            )

    @property
    def CUDNN_MODE(self) -> str:
        r"""
        CuDNN computation mode, should be one of {"repeatability", "performance"}.
        Default value is ``"repeatability"``.

        Note
        ----
        As of now, due to the use of :meth:`torch.Tensor.scatter_add_`
        operation, the results are not completely reproducible even when
        ``CUDNN_MODE`` is set to ``"repeatability"``, if GPU is used as
        computation device. Exact repeatability can only be achieved on CPU.
        The situtation might change with new releases of :mod:`torch`.
        """
        return self._CUDNN_MODE

    @CUDNN_MODE.setter
    def CUDNN_MODE(self, cudnn_mode: str) -> None:
        if cudnn_mode not in ("repeatability", "performance"):
            raise ValueError("Invalid mode!")
        self._CUDNN_MODE = cudnn_mode
        torch.backends.cudnn.deterministic = self._CUDNN_MODE == "repeatability"
        torch.backends.cudnn.benchmark = self._CUDNN_MODE == "performance"

    @property
    def MASKED_GPUS(self) -> list[int]:
        r"""
        A list of GPUs that should not be used when selecting computation device.
        This must be set before initializing any model, otherwise would be ineffective.
        Default value is ``[]``.
        """
        return self._MASKED_GPUS

    @MASKED_GPUS.setter
    def MASKED_GPUS(self, masked_gpus: list[int]) -> None:
        if masked_gpus:
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for item in masked_gpus:
                if item >= device_count:
                    raise ValueError(f'GPU device "{item}" is non-existent!')
        self._MASKED_GPUS = masked_gpus

    @property
    def ARRAY_SHUFFLE_NUM_WORKERS(self) -> int:
        r"""
        Number of background workers for array data shuffling.
        Default value is ``0``.
        """
        return self._ARRAY_SHUFFLE_NUM_WORKERS

    @ARRAY_SHUFFLE_NUM_WORKERS.setter
    def ARRAY_SHUFFLE_NUM_WORKERS(self, array_shuffle_num_workers: int) -> None:
        self._ARRAY_SHUFFLE_NUM_WORKERS = array_shuffle_num_workers

    @property
    def GRAPH_SHUFFLE_NUM_WORKERS(self) -> int:
        r"""
        Number of background workers for graph data shuffling.
        Default value is ``1``.
        """
        return self._GRAPH_SHUFFLE_NUM_WORKERS

    @GRAPH_SHUFFLE_NUM_WORKERS.setter
    def GRAPH_SHUFFLE_NUM_WORKERS(self, graph_shuffle_num_workers: int) -> None:
        self._GRAPH_SHUFFLE_NUM_WORKERS = graph_shuffle_num_workers

    @property
    def FORCE_TERMINATE_WORKER_PATIENCE(self) -> int:
        r"""
        Seconds to wait before force terminating unresponsive workers.
        Default value is ``60``.
        """
        return self._FORCE_TERMINATE_WORKER_PATIENCE

    @FORCE_TERMINATE_WORKER_PATIENCE.setter
    def FORCE_TERMINATE_WORKER_PATIENCE(self, force_terminate_worker_patience: int) -> None:
        self._FORCE_TERMINATE_WORKER_PATIENCE = force_terminate_worker_patience

    @property
    def DATALOADER_NUM_WORKERS(self) -> int:
        r"""
        Number of worker processes to use in data loader.
        Default value is ``0``.
        """
        return self._DATALOADER_NUM_WORKERS

    @DATALOADER_NUM_WORKERS.setter
    def DATALOADER_NUM_WORKERS(self, dataloader_num_workers: int) -> None:
        if dataloader_num_workers > 8:
            self.logger.warning(
                "Worker number 1-8 is generally sufficient, "
                "too many workers might have negative impact on speed."
            )
        self._DATALOADER_NUM_WORKERS = dataloader_num_workers

    @property
    def DATALOADER_FETCHES_PER_WORKER(self) -> int:
        r"""
        Number of fetches per worker per batch to use in data loader.
        Default value is ``4``.
        """
        return self._DATALOADER_FETCHES_PER_WORKER

    @DATALOADER_FETCHES_PER_WORKER.setter
    def DATALOADER_FETCHES_PER_WORKER(self, dataloader_fetches_per_worker: int) -> None:
        self._DATALOADER_FETCHES_PER_WORKER = dataloader_fetches_per_worker

    @property
    def DATALOADER_FETCHES_PER_BATCH(self) -> int:
        r"""
        Number of fetches per batch in data loader (read-only).
        """
        return max(1, self.DATALOADER_NUM_WORKERS) * self.DATALOADER_FETCHES_PER_WORKER

    @property
    def DATALOADER_PIN_MEMORY(self) -> bool:
        r"""
        Whether to use pin memory in data loader.
        Default value is ``True``.
        """
        return self._DATALOADER_PIN_MEMORY

    @DATALOADER_PIN_MEMORY.setter
    def DATALOADER_PIN_MEMORY(self, dataloader_pin_memory: bool):
        self._DATALOADER_PIN_MEMORY = dataloader_pin_memory

    @property
    def CHECKPOINT_SAVE_INTERVAL(self) -> int:
        r"""
        Automatically save checkpoints every n epochs.
        Default value is ``10``.
        """
        return self._CHECKPOINT_SAVE_INTERVAL

    @CHECKPOINT_SAVE_INTERVAL.setter
    def CHECKPOINT_SAVE_INTERVAL(self, checkpoint_save_interval: int) -> None:
        self._CHECKPOINT_SAVE_INTERVAL = checkpoint_save_interval

    @property
    def CHECKPOINT_SAVE_NUMBERS(self) -> int:
        r"""
        Maximal number of checkpoints to preserve at any point.
        Default value is ``3``.
        """
        return self._CHECKPOINT_SAVE_NUMBERS

    @CHECKPOINT_SAVE_NUMBERS.setter
    def CHECKPOINT_SAVE_NUMBERS(self, checkpoint_save_numbers: int) -> None:
        self._CHECKPOINT_SAVE_NUMBERS = checkpoint_save_numbers

    @property
    def PRINT_LOSS_INTERVAL(self) -> int:
        r"""
        Print loss values every n epochs.
        Default value is ``10``.
        """
        return self._PRINT_LOSS_INTERVAL

    @PRINT_LOSS_INTERVAL.setter
    def PRINT_LOSS_INTERVAL(self, print_loss_interval: int) -> None:
        self._PRINT_LOSS_INTERVAL = print_loss_interval

    @property
    def TENSORBOARD_FLUSH_SECS(self) -> int:
        r"""
        Flush tensorboard logs to file every n seconds.
        Default values is ``5``.
        """
        return self._TENSORBOARD_FLUSH_SECS

    @TENSORBOARD_FLUSH_SECS.setter
    def TENSORBOARD_FLUSH_SECS(self, tensorboard_flush_secs: int) -> None:
        self._TENSORBOARD_FLUSH_SECS = tensorboard_flush_secs

    @property
    def ALLOW_TRAINING_INTERRUPTION(self) -> bool:
        r"""
        Allow interruption before model training converges.
        Default values is ``True``.
        """
        return self._ALLOW_TRAINING_INTERRUPTION

    @ALLOW_TRAINING_INTERRUPTION.setter
    def ALLOW_TRAINING_INTERRUPTION(self, allow_training_interruption: bool) -> None:
        self._ALLOW_TRAINING_INTERRUPTION = allow_training_interruption

    @property
    def BEDTOOLS_PATH(self) -> str:
        r"""
        Path to bedtools executable.
        Default value is ``bedtools``.
        """
        return self._BEDTOOLS_PATH

    @BEDTOOLS_PATH.setter
    def BEDTOOLS_PATH(self, bedtools_path: str) -> None:
        self._BEDTOOLS_PATH = bedtools_path
        set_bedtools_path(bedtools_path)


config = ConfigManager()


# ---------------------------- Interruption handling ----------------------------


@logged


# --------------------------- Constrained data frame ----------------------------


@logged
class ConstrainedDataFrame(pd.DataFrame):
    r"""
    Data frame with certain format constraints

    Note
    ----
    Format constraints are checked and maintained automatically.
    """

    def __init__(self, *args, **kwargs) -> None:
        df = pd.DataFrame(*args, **kwargs)
        df = self.rectify(df)
        self.verify(df)
        super().__init__(df)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.verify(self)

    @property
    def _constructor(self) -> type:
        return type(self)

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Rectify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be rectified

        Returns
        -------
        rectified_df
            Rectified data frame
        """
        return df

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        r"""
        Verify data frame for format integrity

        Parameters
        ----------
        df
            Data frame to be verified
        """

    @property
    def df(self) -> pd.DataFrame:
        r"""
        Convert to regular data frame
        """
        return pd.DataFrame(self)

    def __repr__(self) -> str:
        r"""
        Note
        ----
        We need to explicitly call :func:`repr` on the regular data frame
        to bypass integrity verification, because when the terminal is
        too narrow, :mod:`pandas` would split the data frame internally,
        causing format verification to fail.
        """
        return repr(self.df)


# --------------------------- Other utility functions ---------------------------


def get_rs(x: RandomState = None) -> np.random.RandomState:
    r"""
    Get random state object

    Parameters
    ----------
    x
        Object that can be converted to a random state object

    Returns
    -------
    rs
        Random state object
    """
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random


@logged
def tfidf(X: Array) -> Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
    adata: AnnData, n_components: int = 20, use_highly_variable: bool | None = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
