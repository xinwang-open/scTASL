import copy
import functools
import multiprocessing
import operator
import os
import queue
import random
import signal
import uuid
from collections.abc import Mapping
from math import ceil
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import torch
from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset

from .config import (
    config,
    get_default_numpy_dtype,
    get_rs,
    logged,
    processes,
    vertex_degrees,
)
from .typehint import AnyArray, RandomState

DATA_CONFIG = Mapping[str, Any]


def set_seed(seed=1607):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@logged
class Dataset(torch.utils.data.Dataset):
    r"""
    Abstract dataset interface extending that of :class:`torch.utils.data.Dataset`

    Parameters
    ----------
    getitem_size
        Unitary fetch size for each __getitem__ call
    """

    def __init__(self, getitem_size: int = 1) -> None:
        super().__init__()
        self.getitem_size = getitem_size
        self.shuffle_seed: int | None = None
        self.seed_queue: multiprocessing.Queue | None = None
        self.propose_queue: multiprocessing.Queue | None = None
        self.propose_cache: Mapping[int, Any] = {}

    @property
    def has_workers(self) -> bool:
        r"""
        Whether background shuffling workers have been registered
        """
        self_processes = processes[id(self)]
        pl = bool(self_processes)
        sq = self.seed_queue is not None
        pq = self.propose_queue is not None
        if not pl == sq == pq:
            raise RuntimeError("Background shuffling seems broken!")
        return pl and sq and pq

    def prepare_shuffle(self, num_workers: int = 1, random_seed: int = 0) -> None:
        r"""
        Prepare dataset for custom shuffling

        Parameters
        ----------
        num_workers
            Number of background workers for data shuffling
        random_seed
            Initial random seed (will increase by 1 with every shuffle call)
        """
        if self.has_workers:
            self.clean()
        self_processes = processes[id(self)]
        self.shuffle_seed = random_seed
        if num_workers:
            self.seed_queue = multiprocessing.Queue()
            self.propose_queue = multiprocessing.Queue()
            for i in range(num_workers):
                p = multiprocessing.Process(target=self.shuffle_worker)
                p.start()
                self.logger.debug("Started background process: %d", p.pid)
                self_processes[p.pid] = p
                self.seed_queue.put(self.shuffle_seed + i)

    def shuffle(self) -> None:
        r"""
        Custom shuffling
        """
        if self.has_workers:
            self_processes = processes[id(self)]
            self.seed_queue.put(self.shuffle_seed + len(self_processes))  # Look ahead
            while self.shuffle_seed not in self.propose_cache:
                shuffle_seed, shuffled = self.propose_queue.get()
                self.propose_cache[shuffle_seed] = shuffled
            self.accept_shuffle(self.propose_cache.pop(self.shuffle_seed))
        else:
            self.accept_shuffle(self.propose_shuffle(self.shuffle_seed))
        self.shuffle_seed += 1

    def shuffle_worker(self) -> None:
        r"""
        Background shuffle worker
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            seed = self.seed_queue.get()
            if seed is None:
                self.propose_queue.put((None, os.getpid()))
                break
            self.propose_queue.put((seed, self.propose_shuffle(seed)))

    def propose_shuffle(self, seed: int) -> Any:
        r"""
        Propose shuffling using a given random seed

        Parameters
        ----------
        seed
            Random seed

        Returns
        -------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def accept_shuffle(self, shuffled: Any) -> None:
        r"""
        Accept shuffling result

        Parameters
        ----------
        shuffled
            Shuffled result
        """
        raise NotImplementedError  # pragma: no cover

    def clean(self) -> None:
        r"""
        Clean up multi-process resources used in custom shuffling
        """
        self_processes = processes[id(self)]
        if not self.has_workers:
            return
        for _ in self_processes:
            self.seed_queue.put(None)
        self.propose_cache.clear()
        while self_processes:
            try:
                first, second = self.propose_queue.get(
                    timeout=config.FORCE_TERMINATE_WORKER_PATIENCE
                )
            except queue.Empty:
                break
            if first is not None:
                continue
            pid = second
            self_processes[pid].join()
            self.logger.debug("Joined background process: %d", pid)
            del self_processes[pid]
        for pid in list(
            self_processes.keys()
        ):  # If some background processes failed to exit gracefully
            self_processes[pid].terminate()
            self_processes[pid].join()
            self.logger.debug("Terminated background process: %d", pid)
            del self_processes[pid]
        self.propose_queue = None
        self.seed_queue = None

    def __del__(self) -> None:
        self.clean()


class AnnDataset(Dataset):
    r"""
    Dataset for :class:`anndata.AnnData` objects with partial pairing support.

    Parameters
    ----------
    *adatas
        An arbitrary number of configured :class:`anndata.AnnData` objects
    data_configs
        Data configurations, one per dataset
    mode
        Data mode, must be one of ``{"train", "eval"}``
    getitem_size
        Unitary fetch size for each __getitem__ call
    """

    def __init__(
        self,
        adatas: list[AnnData],
        data_configs: list[DATA_CONFIG],
        mode: str = "train",
        getitem_size: int = 1,
    ) -> None:
        # super().__init__()
        super().__init__(getitem_size=getitem_size)
        # self.getitem_size = getitem_size
        if mode not in ("train", "eval"):
            raise ValueError("Invalid `mode`!")
        self.mode = mode
        self.adatas = adatas
        self.data_configs = data_configs

    @property
    def adatas(self) -> list[AnnData]:
        r"""
        Internal :class:`AnnData` objects
        """
        return self._adatas

    @property
    def data_configs(self) -> list[DATA_CONFIG]:
        r"""
        Data configuration for each dataset
        """
        return self._data_configs

    @adatas.setter
    def adatas(self, adatas: list[AnnData]) -> None:
        self.sizes = [adata.shape[0] for adata in adatas]
        if min(self.sizes) == 0:
            raise ValueError("Empty dataset is not allowed!")
        self._adatas = adatas

    @data_configs.setter
    def data_configs(self, data_configs: list[DATA_CONFIG]) -> None:
        if len(data_configs) != len(self.adatas):
            raise ValueError("Number of data configs must match " "the number of datasets!")
        self.data_idx, self.extracted_data = self._extract_data(data_configs)
        self.view_idx = (
            pd.concat([data_idx.to_series() for data_idx in self.data_idx])
            .drop_duplicates()
            .to_numpy()
        )
        self.size = self.view_idx.size
        self.shuffle_idx, self.shuffle_pmsk = self._get_idx_pmsk(self.view_idx)
        self._data_configs = data_configs

    def _get_idx_pmsk(
        self, view_idx: np.ndarray, random_fill: bool = False, random_state: RandomState = None
    ) -> tuple[np.ndarray, np.ndarray]:
        rs = get_rs(random_state) if random_fill else None
        shuffle_idx, shuffle_pmsk = [], []
        for data_idx in self.data_idx:
            idx = data_idx.get_indexer(view_idx)
            pmsk = idx >= 0
            n_true = pmsk.sum()
            n_false = pmsk.size - n_true
            idx[~pmsk] = (
                rs.choice(idx[pmsk], n_false, replace=True)
                if random_fill
                else idx[pmsk][np.mod(np.arange(n_false), n_true)]
            )
            shuffle_idx.append(idx)
            shuffle_pmsk.append(pmsk)
        return np.stack(shuffle_idx, axis=1), np.stack(shuffle_pmsk, axis=1)

    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        s = slice(index * self.getitem_size, min((index + 1) * self.getitem_size, self.size))
        shuffle_idx = self.shuffle_idx[s].T
        shuffle_pmsk = self.shuffle_pmsk[s]
        items = [
            torch.as_tensor(self._index_array(data, idx))
            for extracted_data in self.extracted_data
            for idx, data in zip(shuffle_idx, extracted_data)
        ]
        items.append(torch.as_tensor(shuffle_pmsk))
        return items

    @staticmethod
    def _index_array(arr: AnyArray, idx: np.ndarray) -> np.ndarray:
        if isinstance(arr, (h5py.Dataset, CSRDataset, CSCDataset)):
            rank = scipy.stats.rankdata(idx, method="dense") - 1
            sorted_idx = np.empty(rank.max() + 1, dtype=int)
            sorted_idx[rank] = idx
            arr = arr[sorted_idx.tolist()][rank.tolist()]  # Convert to sequantial access and back
        else:
            arr = arr[idx]
        return arr.toarray() if scipy.sparse.issparse(arr) else arr

    def _extract_data(
        self, data_configs: list[DATA_CONFIG]
    ) -> tuple[
        list[pd.Index], tuple[list[AnyArray], list[AnyArray], list[AnyArray], list[AnyArray]]
    ]:
        if self.mode == "eval":
            return self._extract_data_eval(data_configs)
        return self._extract_data_train(data_configs)  # self.mode == "train"

    def _extract_data_train(
        self, data_configs: list[DATA_CONFIG]
    ) -> tuple[
        list[pd.Index], tuple[list[AnyArray], list[AnyArray], list[AnyArray], list[AnyArray]]
    ]:
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xbch = [
            self._extract_xbch(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xlbl = [
            self._extract_xlbl(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        return xuid, (x, xbch, xlbl)

    def _extract_data_eval(
        self, data_configs: list[DATA_CONFIG]
    ) -> tuple[
        list[pd.Index], tuple[list[AnyArray], list[AnyArray], list[AnyArray], list[AnyArray]]
    ]:
        default_dtype = get_default_numpy_dtype()
        xuid = [
            self._extract_xuid(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        x = [
            self._extract_x(adata, data_config)
            for adata, data_config in zip(self.adatas, data_configs)
        ]
        xbch = xlbl = [np.empty((adata.shape[0], 0), dtype=int) for adata in self.adatas]

        return xuid, (x, xbch, xlbl)

    def _extract_x(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        default_dtype = get_default_numpy_dtype()
        x = adata.X
        if x.dtype.type is not default_dtype:
            if isinstance(x, (h5py.Dataset, CSRDataset, CSCDataset)):
                raise RuntimeError(
                    f"User is responsible for ensuring a {default_dtype} dtype "
                    f"when using backed data!"
                )
            x = x.astype(default_dtype)
        if scipy.sparse.issparse(x):
            x = x.tocsr()
        return x

    def _extract_xbch(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_batch = data_config["use_batch"]
        batches = data_config["batches"]
        if use_batch:
            if use_batch not in adata.obs:
                raise ValueError(
                    f"Configured data batch '{use_batch}' " f"cannot be found in input data!"
                )
            return batches.get_indexer(adata.obs[use_batch])
        return np.zeros(adata.shape[0], dtype=int)

    def _extract_xlbl(self, adata: AnnData, data_config: DATA_CONFIG) -> AnyArray:
        use_cell_type = data_config["use_cell_type"]
        cell_types = data_config["cell_types"]
        if use_cell_type:
            if use_cell_type not in adata.obs:
                raise ValueError(
                    f"Configured cell type '{use_cell_type}' " f"cannot be found in input data!"
                )
            return cell_types.get_indexer(adata.obs[use_cell_type])
        return -np.ones(adata.shape[0], dtype=int)

    def _extract_xuid(self, adata: AnnData, data_config: DATA_CONFIG) -> pd.Index:
        if data_config["use_obs_names"]:
            xuid = adata.obs_names.to_numpy()
        else:  # NOTE: Assuming random UUIDs never collapse with anything
            self.logger.debug("Generating random xuid...")
            xuid = np.array([uuid.uuid4().hex for _ in range(adata.shape[0])])
        if len(set(xuid)) != xuid.size:
            raise ValueError("Non-unique cell ID!")
        return pd.Index(xuid)

    def propose_shuffle(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rs = get_rs(seed)
        view_idx = rs.permutation(self.view_idx)
        return self._get_idx_pmsk(view_idx, random_fill=True, random_state=rs)

    def accept_shuffle(self, shuffled: tuple[np.ndarray, np.ndarray]) -> None:
        self.shuffle_idx, self.shuffle_pmsk = shuffled

    def random_split(
        self, fractions: list[float], random_state: RandomState = None
    ) -> list["AnnDataset"]:
        r"""
        Randomly split the dataset into multiple subdatasets according to
        given fractions.

        Parameters
        ----------
        fractions
            Fraction of each split
        random_state
            Random state

        Returns
        -------
        subdatasets
            A list of splitted subdatasets
        """
        if min(fractions) <= 0:
            raise ValueError("Fractions should be greater than 0!")
        if sum(fractions) != 1:
            raise ValueError("Fractions do not sum to 1!")
        rs = get_rs(random_state)
        cum_frac = np.cumsum(fractions)
        view_idx = rs.permutation(self.view_idx)
        split_pos = np.round(cum_frac * view_idx.size).astype(int)
        split_idx = np.split(view_idx, split_pos[:-1])  # Last pos produces an extra empty split
        subdatasets = []
        for idx in split_idx:
            sub = copy.copy(self)
            sub.view_idx = idx
            sub.size = idx.size
            sub.shuffle_idx, sub.shuffle_pmsk = sub._get_idx_pmsk(
                idx
            )  # pylint: disable=protected-access
            subdatasets.append(sub)
        return subdatasets


class GraphDataset(Dataset):
    r"""
    Dataset for bipartite gene-peak graphs with support for negative sampling.

    This dataset constructs edge triplets from PyG graph data
    and supports weighted negative sampling and efficient batching.

    Parameters
    ----------
    graph_data : torch_geometric.data.Data
        PyG graph object containing:
        - edge_index: [2, num_edges] edge indices
        - edge_attr: [num_edges, 1] edge attributes (distance)
        - gene_names: list of gene names
        - peak_names: list of peak names
    neg_samples : int, optional
        Number of negative samples per positive edge. Default is 1.
    weighted_sampling : bool, optional
        Whether to sample negative edges proportional to node degrees. Default is True.
    deemphasize_loops : bool, optional
        Whether to exclude self-loops when computing vertex degrees. Default is True.
    getitem_size : int, optional
        Number of samples per batch. Default is 1.
    max_neg_retries : int, optional
        Maximum retries for negative sampling collision resolution. Default is 10.
    """

    # Distance bins and corresponding weights for edge weighting
    _DIST_BINS = np.array([0, 1, 48_001, 150_001, 250_001])
    _DIST_WEIGHTS = np.array([1.0, 0.8, 0.5, 0.3, 0.1])

    def __init__(
        self,
        graph_data,
        neg_samples: int = 1,
        weighted_sampling: bool = True,
        deemphasize_loops: bool = True,
        getitem_size: int = 1,
        max_neg_retries: int = 10,
    ) -> None:
        super().__init__(getitem_size=getitem_size)

        self.getitem_size = getitem_size
        self.neg_samples = neg_samples
        self.max_neg_retries = max_neg_retries

        # ============================================================
        # Extract raw arrays from PyG graph object (no intermediate sparse matrix)
        # ============================================================
        n_genes = len(graph_data.gene_names)
        n_peaks = len(graph_data.peak_names)

        edge_index = graph_data.edge_index.numpy()  # [2, num_edges]
        edge_attr = graph_data.edge_attr.numpy().ravel()  # [num_edges]

        rows = edge_index[0]  # gene indices
        cols = edge_index[1] - n_genes  # peak indices (remove offset)

        # ============================================================
        # Build triplets directly from raw arrays
        # ============================================================
        self.eidx, self.ewt = self._build_triplets(rows, cols, edge_attr, n_genes, n_peaks)

        # Use explicit node count to handle isolated nodes correctly
        self.vnum = n_genes + n_peaks

        # ============================================================
        # Vertex sampling probabilities
        # ============================================================
        if weighted_sampling:
            if deemphasize_loops:
                mask = self.eidx[0] != self.eidx[1]
            else:
                mask = np.ones(self.ewt.shape, dtype=bool)
            eidx_nl = self.eidx[:, mask]
            ewt_nl = self.ewt[mask]
            degree = vertex_degrees(eidx_nl, ewt_nl, vnum=self.vnum)
        else:
            degree = np.ones(self.vnum, dtype=self.ewt.dtype)

        deg_sum = degree.sum()
        self.vprob = (
            degree / deg_sum
            if deg_sum > 0
            else np.full(self.vnum, 1.0 / self.vnum, dtype=self.ewt.dtype)
        )

        # ============================================================
        # Edge sampling probabilities
        # ============================================================
        total_ewt = float(self.ewt.sum())
        self.eprob = self.ewt / total_ewt
        self.effective_enum = int(round(total_ewt))

        # Total samples including negatives
        self.size = self.effective_enum * (1 + self.neg_samples)

        # Pre-compute edge existence set for fast negative sampling
        self._exist_set = self._build_edge_set(self.eidx)

        # Placeholders for shuffled data
        self.samp_eidx = None
        self.samp_ewt = None

    # ================================================================
    # Triplet construction
    # ================================================================
    @classmethod
    def _dist_to_weights(cls, dists: np.ndarray) -> np.ndarray:
        """Map distances to weights using binned lookup (vectorized)."""
        indices = np.digitize(dists, cls._DIST_BINS) - 1
        return cls._DIST_WEIGHTS[indices]

    @staticmethod
    def _build_triplets(
        rows: np.ndarray,
        cols: np.ndarray,
        dists: np.ndarray,
        n_genes: int,
        n_peaks: int,
    ) -> tuple:
        """
        Build symmetric edge triplets with self-loops from raw arrays.

        Skips intermediate sparse matrix construction for efficiency.
        """
        num_nodes = n_genes + n_peaks

        # Map distances to weights
        weights = GraphDataset._dist_to_weights(dists)

        # Forward edges: gene -> peak (peak indices offset by n_genes)
        src_fwd = rows
        dst_fwd = cols + n_genes

        # Reverse edges: peak -> gene
        src_rev = dst_fwd
        dst_rev = src_fwd

        # Self-loops for all nodes
        loop_idx = np.arange(num_nodes)

        # Concatenate everything at once
        src_all = np.concatenate([src_fwd, src_rev, loop_idx])
        dst_all = np.concatenate([dst_fwd, dst_rev, loop_idx])
        ewt_all = np.concatenate([weights, weights.copy(), np.ones(num_nodes)])

        eidx = np.vstack([src_all, dst_all])
        return eidx, ewt_all

    # ================================================================
    # Negative sampling helpers
    # ================================================================
    @staticmethod
    def _build_edge_set(eidx: np.ndarray) -> set:
        """Encode edge pairs as single int64 for O(1) lookup."""
        vnum = int(eidx.max()) + 1
        codes = eidx[0].astype(np.int64) * vnum + eidx[1].astype(np.int64)
        return set(codes.tolist())

    def _encode_pairs(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Encode (src, dst) pairs into single int64 codes."""
        return src.astype(np.int64) * self.vnum + dst.astype(np.int64)

    def _collision_mask(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Vectorized check for edge existence using pre-computed set."""
        codes = self._encode_pairs(src, dst)
        # Use numpy isin with pre-computed array for large sets,
        # or Python set lookup for moderate sizes
        if len(self._exist_set) > 100_000:
            exist_arr = np.array(list(self._exist_set), dtype=np.int64)
            return np.isin(codes, exist_arr)
        else:
            return np.array([c in self._exist_set for c in codes], dtype=bool)

    # ================================================================
    # Dataset interface
    # ================================================================
    def __len__(self) -> int:
        return ceil(self.size / self.getitem_size)

    def __getitem__(self, index: int):
        if self.samp_eidx is None:
            raise RuntimeError("Call accept_shuffle() before fetching items.")

        start = index * self.getitem_size
        end = min(start + self.getitem_size, self.size)

        return [
            torch.as_tensor(self.samp_eidx[:, start:end], dtype=torch.long),
            torch.as_tensor(self.samp_ewt[start:end], dtype=torch.float32),
        ]

    def propose_shuffle(self, seed: int) -> tuple:
        rs = get_rs(seed)

        # ---- Positive edges ----
        pos_idx = rs.choice(self.ewt.size, self.effective_enum, replace=True, p=self.eprob)
        pi, pj = self.eidx[:, pos_idx]
        pw = np.ones(pos_idx.size, dtype=self.ewt.dtype)

        # ---- Negative sampling (vectorized with bounded retries) ----
        ni = np.repeat(pi, self.neg_samples)
        nj = rs.choice(self.vnum, ni.size, replace=True, p=self.vprob)
        nw = np.zeros(ni.size, dtype=self.ewt.dtype)

        # Resolve collisions with existing edges
        mask = self._collision_mask(ni, nj)
        for _ in range(self.max_neg_retries):
            n_collisions = mask.sum()
            if n_collisions == 0:
                break
            nj[mask] = rs.choice(self.vnum, n_collisions, replace=True, p=self.vprob)
            mask[mask] = self._collision_mask(ni[mask], nj[mask])

        # ---- Concatenate and shuffle ----
        eidx_all = np.vstack(
            [
                np.concatenate([pi, ni]),
                np.concatenate([pj, nj]),
            ]
        )
        ewt_all = np.concatenate([pw, nw])

        perm = rs.permutation(ewt_all.size)
        return eidx_all[:, perm], ewt_all[perm]

    def accept_shuffle(self, shuffled: tuple) -> None:
        self.samp_eidx, self.samp_ewt = shuffled


class DataLoader(torch.utils.data.DataLoader):
    r"""
    Custom data loader that manually shuffles the internal dataset before each
    round of iteration (see :class:`torch.utils.data.DataLoader` for usage)
    """

    def __init__(self, dataset: Dataset, **kwargs) -> None:
        super().__init__(dataset, **kwargs)
        self.collate_fn = (
            self._collate_graph if isinstance(dataset, GraphDataset) else self._collate
        )
        self.shuffle = kwargs["shuffle"] if "shuffle" in kwargs else False

    def __iter__(self) -> "DataLoader":
        if self.shuffle:
            self.dataset.shuffle()  # Customized shuffling
        return super().__iter__()

    @staticmethod
    def _collate(batch):
        return tuple(map(lambda x: torch.cat(x, dim=0), zip(*batch)))

    @staticmethod
    def _collate_graph(batch):
        eidx, ewt = zip(*batch)
        eidx = torch.cat(eidx, dim=1)
        ewt = torch.cat(ewt, dim=0)
        return eidx, ewt


class ParallelDataLoader:
    r"""
    Parallel data loader

    Parameters
    ----------
    *data_loaders
        An arbitrary number of data loaders
    cycle_flags
        Whether each data loader should be cycled in case they are of
        different lengths, by default none of them are cycled.
    """

    def __init__(self, *data_loaders: DataLoader, cycle_flags: list[bool] | None = None) -> None:
        cycle_flags = cycle_flags or [False] * len(data_loaders)
        if len(cycle_flags) != len(data_loaders):
            raise ValueError("Invalid cycle flags!")
        self.cycle_flags = cycle_flags
        self.data_loaders = list(data_loaders)
        self.num_loaders = len(self.data_loaders)
        self.iterators = None

    def __iter__(self) -> "ParallelDataLoader":
        self.iterators = [iter(loader) for loader in self.data_loaders]
        return self

    def _next(self, i: int) -> list[torch.Tensor]:
        try:
            return next(self.iterators[i])
        except StopIteration as e:
            if self.cycle_flags[i]:
                self.iterators[i] = iter(self.data_loaders[i])
                return next(self.iterators[i])
            raise e

    def __next__(self) -> list[torch.Tensor]:
        return functools.reduce(operator.add, [self._next(i) for i in range(self.num_loaders)])
