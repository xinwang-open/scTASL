import collections
import os
from collections.abc import Callable, Mapping
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import pybedtools
import scipy
import torch
from numpy.random import RandomState
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
from statsmodels.stats.multitest import fdrcorrection
from torch_geometric.data import Data
from tqdm import tqdm

from ..common.config import ConstrainedDataFrame


class Bed(ConstrainedDataFrame):
    r"""
    BED format data frame
    """

    COLUMNS = pd.Index(
        [
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "score",
            "strand",
            "thickStart",
            "thickEnd",
            "itemRgb",
            "blockCount",
            "blockSizes",
            "blockStarts",
        ]
    )

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super().rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.loc[:, COLUMNS]

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super().verify(df)
        if len(df.columns) != len(cls.COLUMNS) or np.any(df.columns != cls.COLUMNS):
            raise ValueError("Invalid BED format!")

    @classmethod
    def read_bed(cls, fname: os.PathLike) -> "Bed":
        r"""
        Read BED file

        Parameters
        ----------
        fname
            BED file

        Returns
        -------
        bed
            Loaded :class:`Bed` object
        """
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[: loaded.shape[1]]
        return cls(loaded)

    def write_bed(self, fname: os.PathLike, ncols: int | None = None) -> None:
        r"""
        Write BED file

        Parameters
        ----------
        fname
            BED file
        ncols
            Number of columns to write (by default write all columns)
        """
        if ncols and ncols < 3:
            raise ValueError("`ncols` must be larger than 3!")
        df = self.df.iloc[:, :ncols] if ncols else self
        df.to_csv(fname, sep="\t", header=False, index=False)

    def to_bedtool(self) -> pybedtools.BedTool:
        r"""
        Convert to a :class:`pybedtools.BedTool` object

        Returns
        -------
        bedtool
            Converted :class:`pybedtools.BedTool` object
        """
        return BedTool(
            Interval(
                row["chrom"],
                row["chromStart"],
                row["chromEnd"],
                name=row["name"],
                score=row["score"],
                strand=row["strand"],
            )
            for _, row in self.iterrows()
        )

    def nucleotide_content(self, fasta: os.PathLike) -> pd.DataFrame:
        r"""
        Compute nucleotide content in the BED regions

        Parameters
        ----------
        fasta
            Genomic sequence file in FASTA format

        Returns
        -------
        nucleotide_stat
            Data frame containing nucleotide content statistics for each region
        """
        result = self.to_bedtool().nucleotide_content(
            fi=os.fspath(fasta), s=True
        )  # pylint: disable=unexpected-keyword-arg
        result = pd.DataFrame(
            np.stack([interval.fields[6:15] for interval in result]),
            columns=[
                r"%AT",
                r"%GC",
                r"#A",
                r"#C",
                r"#G",
                r"#T",
                r"#N",
                r"#other",
                r"length",
            ],
        ).astype(
            {
                r"%AT": float,
                r"%GC": float,
                r"#A": int,
                r"#C": int,
                r"#G": int,
                r"#T": int,
                r"#N": int,
                r"#other": int,
                r"length": int,
            }
        )
        pybedtools.cleanup()
        return result

    def strand_specific_start_site(self) -> "Bed":
        r"""
        Convert to strand-specific start sites of genomic features

        Returns
        -------
        start_site_bed
            A new :class:`Bed` object, containing strand-specific start sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return type(self)(df)

    def strand_specific_end_site(self) -> "Bed":
        r"""
        Convert to strand-specific end sites of genomic features

        Returns
        -------
        end_site_bed
            A new :class:`Bed` object, containing strand-specific end sites
            of the current :class:`Bed` object
        """
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromStart"] = df.loc[pos_strand, "chromEnd"] - 1
        df.loc[neg_strand, "chromEnd"] = df.loc[neg_strand, "chromStart"] + 1
        return type(self)(df)

    def expand(
        self,
        upstream: int,
        downstream: int,
        chr_len: Mapping[str, int] | None = None,
    ) -> "Bed":
        r"""
        Expand genomic features towards upstream and downstream

        Parameters
        ----------
        upstream
            Number of bps to expand in the upstream direction
        downstream
            Number of bps to expand in the downstream direction
        chr_len
            Length of each chromosome

        Returns
        -------
        expanded_bed
            A new :class:`Bed` object, containing expanded features
            of the current :class:`Bed` object

        Note
        ----
        Starting position < 0 after expansion is always trimmed.
        Ending position exceeding chromosome length is trimmed only if
        ``chr_len`` is specified.
        """
        if upstream == downstream == 0:
            return self
        df = pd.DataFrame(self, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return type(self)(df)


def get_rs(random_state: RandomState = None) -> RandomState:
    """Convert user input into a numpy RandomState."""
    if random_state is None:
        return np.random.RandomState()
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


def regulatory_inference(
    features: pd.Index,
    feature_embeddings: np.ndarray | list[np.ndarray],
    prior_graph: Data,
    alternative: str = "two.sided",
    random_state: RandomState = None,
    keep_distance: bool = True,
) -> Data:
    r"""
    Regulatory inference based on feature embeddings (PyG Data version)

    Notes
    -----
    - The prior_graph is a PyG Data object:
        Data(edge_index=[2, E], edge_attr=[E, 1], num_nodes=N, node_names=[N], ...)
    - Node alignment is performed using `prior_graph.node_names`:
        v[i] corresponds to the node name prior_graph.node_names[i]
    - Edge scores are computed only on edges in the prior_graph.
    - The null distribution is built by permuting feature identities (row permutation) per embedding model.

    Parameters
    ----------
    features
        Feature names (pd.Index). Must contain all prior_graph.node_names.
    feature_embeddings
        A single embedding array [n_features, d] or a list of arrays (from multiple models).
    prior_graph
        PyG Data graph providing candidate edges (edge_index) and optional distances (edge_attr).
    alternative
        Alternative hypothesis, must be one of {"two.sided", "less", "greater"}.
    random_state
        Random state for permutation.
    keep_distance
        If True and prior_graph.edge_attr exists, keep it in output as the first column ("distance").

    Returns
    -------
    regulatory_graph
        PyG Data with:
        - edge_index: same as prior_graph
        - edge_attr: [E, k] columns = [distance(optional), score, pval, qval]
        - edge_attr_dict: dict containing numpy arrays for "distance"(optional), "score", "pval", "qval"
        - node metadata copied from the prior_graph when present
    """
    # ---- normalize feature_embeddings input ----
    if isinstance(feature_embeddings, np.ndarray):
        feature_embeddings = [feature_embeddings]

    n_features = set(item.shape[0] for item in feature_embeddings)
    if len(n_features) != 1:
        raise ValueError("All feature embeddings must have the same number of rows!")
    if n_features.pop() != features.shape[0]:
        raise ValueError("Feature embeddings do not match the number of feature names!")

    # ---- validate prior_graph fields ----
    if not hasattr(prior_graph, "edge_index") or prior_graph.edge_index is None:
        raise ValueError("PyG prior_graph must contain `edge_index`!")
    if not hasattr(prior_graph, "node_names") or prior_graph.node_names is None:
        raise ValueError("PyG prior_graph must contain `node_names` for name-based alignment!")

    edge_index = prior_graph.edge_index
    if not torch.is_tensor(edge_index):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    else:
        edge_index = edge_index.long()
    edge_index_cpu = edge_index.detach().cpu()

    # Number of nodes
    if hasattr(prior_graph, "num_nodes") and prior_graph.num_nodes is not None:
        n_nodes = int(prior_graph.num_nodes)
    else:
        n_nodes = (
            int(edge_index_cpu.max().item()) + 1
            if edge_index_cpu.numel() > 0
            else len(prior_graph.node_names)
        )

    node_names = pd.Index(list(prior_graph.node_names))
    if node_names.shape[0] != n_nodes:
        raise ValueError(f"`node_names` length ({node_names.shape[0]}) != num_nodes ({n_nodes})")

    # ---- align embeddings to prior_graph node order using names ----
    node_idx = features.get_indexer(node_names)
    if (node_idx < 0).any():
        missing = node_names[node_idx < 0].tolist()
        raise ValueError(f"Some prior_graph nodes are not found in `features`: {missing[:10]} ...")

    # Reorder feature embeddings so that row i matches prior_graph node i
    emb_aligned = [item[node_idx] for item in feature_embeddings]

    # ---- build permuted background and normalize vectors (cosine similarity) ----
    rs = get_rs(random_state)

    vperm = np.stack([rs.permutation(item) for item in emb_aligned], axis=1)  # [N, M, d]
    vperm = vperm / np.linalg.norm(vperm, axis=-1, keepdims=True)

    v = np.stack(emb_aligned, axis=1)  # [N, M, d]
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)

    # ---- compute foreground score and background samples on prior_graph edges ----
    src = edge_index_cpu[0].numpy()
    tgt = edge_index_cpu[1].numpy()

    fg: list[float] = []
    bg: list[np.ndarray] = []

    for s, t in tqdm(
        zip(src, tgt),
        total=src.shape[0],
        desc="regulatory_inference",
    ):
        # Foreground: mean cosine similarity across models
        fg.append((v[s] * v[t]).sum(axis=1).mean())

        # Background: cosine similarity under permuted feature identities
        bg.append((vperm[s] * vperm[t]).sum(axis=1))

    fg = np.asarray(fg, dtype=np.float64)

    # ---- empirical p-values from pooled background distribution ----
    bg = np.sort(np.concatenate(bg).astype(np.float64))
    quantile = np.searchsorted(bg, fg, side="left") / bg.size

    if alternative == "two.sided":
        pval = 2 * np.minimum(quantile, 1 - quantile)
    elif alternative == "greater":
        pval = 1 - quantile
    elif alternative == "less":
        pval = quantile
    else:
        raise ValueError("Unrecognized `alternative`!")

    qval = fdrcorrection(pval)[1].astype(np.float64)

    # ---- pack output edge attributes ----
    cols = []
    edge_attr_dict = {}

    # Keep the original prior distance (e.g., genomic distance) if requested
    if keep_distance and hasattr(prior_graph, "edge_attr") and prior_graph.edge_attr is not None:
        dist = prior_graph.edge_attr
        if not torch.is_tensor(dist):
            dist = torch.as_tensor(dist, dtype=torch.float32)
        dist = dist.float()
        if dist.ndim == 1:
            dist = dist[:, None]
        cols.append(dist)
        edge_attr_dict["distance"] = dist.detach().cpu().numpy().reshape(-1)

    score_t = torch.from_numpy(fg.astype(np.float32))[:, None]
    pval_t = torch.from_numpy(pval.astype(np.float32))[:, None]
    qval_t = torch.from_numpy(qval.astype(np.float32))[:, None]

    cols.extend([score_t, pval_t, qval_t])
    edge_attr_out = torch.cat(cols, dim=1)  # [E, k]

    edge_attr_dict["score"] = fg
    edge_attr_dict["pval"] = pval
    edge_attr_dict["qval"] = qval

    out = Data(
        edge_index=prior_graph.edge_index,
        edge_attr=edge_attr_out,
        num_nodes=n_nodes,
    )

    # Copy node/graph metadata
    out.node_names = list(prior_graph.node_names)
    if hasattr(prior_graph, "node_type"):
        out.node_type = prior_graph.node_type
    if hasattr(prior_graph, "gene_names"):
        out.gene_names = prior_graph.gene_names
    if hasattr(prior_graph, "peak_names"):
        out.peak_names = prior_graph.peak_names

    out.edge_attr_dict = edge_attr_dict
    return out


def _as_bool_mask(x, length: int, device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        t = torch.as_tensor(x, dtype=torch.bool, device=device)
    else:
        t = x.to(device=device, dtype=torch.bool)
    if t.numel() != length:
        raise ValueError(f"Mask length mismatch: expected {length}, got {t.numel()}")
    return t


def _infer_gene_peak_masks(
    node_names: np.ndarray,
    node_type: np.ndarray | None,
    gene_names: list | np.ndarray | None = None,
    peak_names: list | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer which nodes are genes / peaks.

    Priority:
      1) node_type if present (supports str or int codes)
      2) fallback to membership in gene_names / peak_names sets
    """
    n = node_names.shape[0]
    gene_mask = np.zeros(n, dtype=bool)
    peak_mask = np.zeros(n, dtype=bool)

    if node_type is not None:
        nt = np.asarray(node_type)

        # string types
        if nt.dtype.kind in ("U", "S", "O"):
            nt_low = np.char.lower(nt.astype(str))
            gene_mask = nt_low == "gene"
            peak_mask = nt_low == "peak"
            return gene_mask, peak_mask

        # integer-coded types: guess from membership if possible
        if gene_names is not None or peak_names is not None:
            gset = set(map(str, gene_names)) if gene_names is not None else set()
            pset = set(map(str, peak_names)) if peak_names is not None else set()
            name_str = node_names.astype(str)

            gm = (
                np.fromiter((nm in gset for nm in name_str), count=n, dtype=bool)
                if gset
                else np.zeros(n, bool)
            )
            pm = (
                np.fromiter((nm in pset for nm in name_str), count=n, dtype=bool)
                if pset
                else np.zeros(n, bool)
            )

            codes = np.unique(nt)
            for c in codes:
                idx = nt == c
                if idx.sum() == 0:
                    continue
                if gm[idx].mean() >= pm[idx].mean():
                    gene_mask[idx] = True
                else:
                    peak_mask[idx] = True
            return gene_mask, peak_mask

        # last resort: 0 -> gene, else -> peak
        gene_mask = nt == 0
        peak_mask = nt != 0
        return gene_mask, peak_mask

    # no node_type: use membership
    gset = set(map(str, gene_names)) if gene_names is not None else set()
    pset = set(map(str, peak_names)) if peak_names is not None else set()
    name_str = node_names.astype(str)

    if gset:
        gene_mask = np.fromiter((nm in gset for nm in name_str), count=n, dtype=bool)
    if pset:
        peak_mask = np.fromiter((nm in pset for nm in name_str), count=n, dtype=bool)

    return gene_mask, peak_mask


def edge_subgraph(
    g: Data,
    edge_mask: np.ndarray | torch.Tensor,
    drop_isolated: bool = True,
    reindex_nodes: bool = True,
    update_gene_peak_names: bool = True,
) -> Data:
    """
    Edge-induced subgraph with consistent metadata updates.

    Output attribute order is fixed to:
      edge_index, edge_attr, num_nodes, node_names, node_type, gene_names, peak_names, edge_attr_dict(last)
    """
    if not hasattr(g, "edge_index") or g.edge_index is None:
        raise ValueError("Input graph must have `edge_index`.")

    ei = g.edge_index
    if not torch.is_tensor(ei):
        ei = torch.as_tensor(ei, dtype=torch.long)
    else:
        ei = ei.long()

    E = ei.size(1)
    num_nodes = int(getattr(g, "num_nodes", 0) or 0)
    if num_nodes <= 0:
        num_nodes = int(ei.max().item()) + 1 if ei.numel() > 0 else 0

    em = _as_bool_mask(edge_mask, E, ei.device)

    # ---- filter edges ----
    ei_f = ei[:, em]

    ea_f = None
    if hasattr(g, "edge_attr") and g.edge_attr is not None:
        ea = g.edge_attr
        if not torch.is_tensor(ea):
            ea = torch.as_tensor(ea)
        ea_f = ea[em]

    ead_f: dict[str, np.ndarray] | None = None
    if hasattr(g, "edge_attr_dict") and g.edge_attr_dict is not None:
        keep_np = em.detach().cpu().numpy()
        ead_f = {k: np.asarray(v)[keep_np] for k, v in g.edge_attr_dict.items()}

    # ---- compute node subset if needed ----
    node_names_sub = None
    node_type_sub = None
    gene_names_sub = None
    peak_names_sub = None

    if not drop_isolated:
        if hasattr(g, "node_names"):
            node_names_sub = np.asarray(list(g.node_names), dtype=object)
        if hasattr(g, "node_type"):
            node_type_sub = np.asarray(g.node_type)
        gene_names_sub = list(getattr(g, "gene_names", [])) if hasattr(g, "gene_names") else []
        peak_names_sub = list(getattr(g, "peak_names", [])) if hasattr(g, "peak_names") else []
        out_num_nodes = num_nodes

    else:
        if ei_f.numel() == 0:
            node_names_sub = np.asarray([], dtype=object)
            node_type_sub = np.asarray([], dtype=object) if hasattr(g, "node_type") else None
            gene_names_sub = []
            peak_names_sub = []
            out_num_nodes = 0 if reindex_nodes else num_nodes
        else:
            used = torch.unique(ei_f.reshape(-1))
            used, _ = torch.sort(used)
            used_np = used.detach().cpu().numpy()

            if hasattr(g, "node_names"):
                node_names_sub = np.asarray(list(g.node_names), dtype=object)[used_np]
            if hasattr(g, "node_type"):
                node_type_sub = np.asarray(g.node_type)[used_np]

            if update_gene_peak_names and node_names_sub is not None:
                gm, pm = _infer_gene_peak_masks(
                    node_names=node_names_sub,
                    node_type=node_type_sub,
                    gene_names=getattr(g, "gene_names", None),
                    peak_names=getattr(g, "peak_names", None),
                )
                gene_names_sub = node_names_sub[gm].tolist()
                peak_names_sub = node_names_sub[pm].tolist()
            else:
                gene_names_sub = (
                    list(getattr(g, "gene_names", [])) if hasattr(g, "gene_names") else []
                )
                peak_names_sub = (
                    list(getattr(g, "peak_names", [])) if hasattr(g, "peak_names") else []
                )

            # reindex nodes
            if reindex_nodes:
                new_id = -torch.ones((num_nodes,), dtype=torch.long, device=ei.device)
                new_id[used] = torch.arange(used.numel(), device=ei.device)
                ei_f = new_id[ei_f]
                out_num_nodes = int(used.numel())
            else:
                out_num_nodes = num_nodes

    # ---- build output with FIXED attribute insertion order ----
    out = Data()

    # 1) edge_index
    out.edge_index = ei_f
    # 2) edge_attr
    out.edge_attr = ea_f
    # 3) num_nodes
    out.num_nodes = int(out_num_nodes)

    # 4) node_names
    if node_names_sub is not None and hasattr(g, "node_names"):
        out.node_names = node_names_sub.tolist()
    # 5) node_type
    if node_type_sub is not None and hasattr(g, "node_type"):
        out.node_type = node_type_sub

    # 6) gene_names
    if hasattr(g, "gene_names"):
        out.gene_names = gene_names_sub
    # 7) peak_names
    if hasattr(g, "peak_names"):
        out.peak_names = peak_names_sub

    # 8) edge_attr_dict (ALWAYS LAST)
    if ead_f is not None:
        out.edge_attr_dict = ead_f

    return out


def interval_dist(x: Interval, y: Interval) -> int:
    r"""
    Compute distance and relative position between two bed intervals

    Parameters
    ----------
    x
        First interval
    y
        Second interval

    Returns
    -------
    dist
        Signed distance between ``x`` and ``y``
    """
    if x.chrom != y.chrom:
        return np.inf * (-1 if x.chrom < y.chrom else 1)
    if x.start < y.stop and y.start < x.stop:
        return 0
    if x.stop <= y.start:
        return x.stop - y.start - 1
    if y.stop <= x.start:
        return x.start - y.stop + 1


def window_graph(
    left: Bed | str,
    right: Bed | str,
    window_size: int,
    left_sorted: bool = False,
    right_sorted: bool = False,
    attr_fn: Callable[[Interval, Interval, float], Mapping[str, Any]] | None = None,
) -> Data:
    r"""
    Construct a window graph between two sets of genomic features using PyG.

    Features pairs within *window_size* bp are connected by directed edges.

    Parameters
    ----------
    left
        First feature set, either a :class:`Bed` object or path to a bed file.
    right
        Second feature set, either a :class:`Bed` object or path to a bed file.
    window_size
        Window size (in bp).
    left_sorted
        Whether ``left`` is already sorted.
    right_sorted
        Whether ``right`` is already sorted.
    attr_fn
        Function to compute edge attributes for connected features.
        Should accept (l, r, d) and return a ``dict[str, scalar | tensor]``.
        All returned dicts **must** have the same keys.

    Returns
    -------
    torch_geometric.data.Data
        A PyG Data object containing:

        - ``edge_index`` – ``[2, E]`` LongTensor of directed edges
        - ``edge_attr``  – ``dict`` of ``{key: Tensor[E, ...]}`` (if *attr_fn* provided)
        - ``node_names`` – ``list[str]`` mapping node index → name
        - ``name_to_idx``– ``dict[str, int]`` mapping name → node index
    """

    # ── Input preprocessing (identical to original) ──────────────────────
    if isinstance(left, Bed):
        pbar_total = len(left)
        left = left.to_bedtool()
    else:
        pbar_total = None
        left = pybedtools.BedTool(left)
    if not left_sorted:
        left = left.sort(stream=True)
    left = iter(left)

    if isinstance(right, Bed):
        right = right.to_bedtool()
    else:
        right = pybedtools.BedTool(right)
    if not right_sorted:
        right = right.sort(stream=True)
    right = iter(right)

    attr_fn = attr_fn or (lambda l, r, d: {})

    if pbar_total is not None:
        left = tqdm(left, total=pbar_total, desc="window_graph")

    # ── Name → index mapping ─────────────────────────────────────────────
    name_to_idx: dict[str, int] = {}
    node_names: list[str] = []

    def _get_idx(name: str) -> int:
        """Return existing index or assign a new one."""
        idx = name_to_idx.get(name)
        if idx is None:
            idx = len(node_names)
            name_to_idx[name] = idx
            node_names.append(name)
        return idx

    # ── Edge collection ──────────────────────────────────────────────────
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_attr_list: list[Mapping[str, Any]] = []

    def _add_edge(l: Interval, r: Interval, d: float) -> None:
        src_list.append(_get_idx(l.name))
        dst_list.append(_get_idx(r.name))
        attrs = attr_fn(l, r, d)
        if attrs:
            edge_attr_list.append(attrs)

    # ── Sliding window (identical logic to original) ─────────────────────
    window = collections.OrderedDict()  # ordered set of right intervals

    for l in left:
        for r in list(window.keys()):
            d = interval_dist(l, r)
            if -window_size <= d <= window_size:
                _add_edge(l, r, d)
            elif d > window_size:
                del window[r]
            else:  # d < -window_size
                break
        else:
            for r in right:
                d = interval_dist(l, r)
                if -window_size <= d <= window_size:
                    _add_edge(l, r, d)
                elif d > window_size:
                    continue
                window[r] = None
                if d < -window_size:
                    break

    pybedtools.cleanup()

    # ── Build PyG Data object ────────────────────────────────────────────
    num_nodes = len(node_names)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    # Stack edge attributes and store in edge_attr_dict so edge_subgraph propagates them
    if edge_attr_list:
        keys = edge_attr_list[0].keys()
        ead: dict[str, np.ndarray] = {}
        for k in keys:
            values = [ea[k] for ea in edge_attr_list]
            if isinstance(values[0], torch.Tensor):
                ead[k] = torch.stack(values).numpy()
            else:
                ead[k] = np.array(values, dtype=np.float32)
        data.edge_attr_dict = ead

    # Attach name mappings for convenience
    data.node_names = node_names  # type: ignore[attr-defined]
    data.name_to_idx = name_to_idx  # type: ignore[attr-defined]

    return data


def write_links(
    graph: Data,
    rna_var: pd.DataFrame,
    atac_var: pd.DataFrame,
    file: os.PathLike,
    gene_name_col: str = "gene_name",
    peak_name_col: str = "peak_name",
    keep_attrs: list[str] | None = None,
    gene_as_start_site: bool = True,
) -> None:
    r"""
    Export a PyG regulatory graph into a links file using rna.var / atac.var inputs.

    Parameters
    ----------
    graph
        PyG Data regulatory graph. Must contain:
        - edge_index: [2, E]
        - node_names: [N]
        Optionally:
        - edge_attr_dict: dict[str, array-like], each of length E
    rna_var
        RNA AnnData.var-like table containing gene coordinates.
        Must contain 'chrom', 'chromStart', 'chromEnd' and a gene name column.
    atac_var
        ATAC AnnData.var-like table containing peak coordinates.
        Must contain 'chrom', 'chromStart', 'chromEnd' and a peak name column.
    file
        Output file path.
    gene_name_col
        Column in rna_var that matches gene node names used in graph.node_names.
    peak_name_col
        Column in atac_var that matches peak node names used in graph.node_names.
    keep_attrs
        Edge attributes to keep (e.g., ["distance", "score", "pval", "qval"]).
        Values are read from `graph.edge_attr_dict`.
    gene_as_start_site
        If True, convert gene intervals into strand-specific start sites (TSS-like, 1 bp).
        This requires rna_var['strand'] to be '+'/'-'. If not strand-specific, set False.
    """
    keep_attrs = keep_attrs or []

    # ---- basic checks ----
    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        raise ValueError("`graph` must contain `edge_index`!")
    if not hasattr(graph, "node_names") or graph.node_names is None:
        raise ValueError("`graph` must contain `node_names`!")
    if gene_name_col not in rna_var.columns:
        raise ValueError(f"`rna_var` missing gene_name_col: {gene_name_col}")
    if peak_name_col not in atac_var.columns:
        raise ValueError(f"`atac_var` missing peak_name_col: {peak_name_col}")

    # ---- build Bed(source) from rna.var (name = gene_name) ----
    rna_df = rna_var.copy()
    rna_df["name"] = rna_df[gene_name_col].astype(str)
    source_bed = Bed(rna_df)
    if gene_as_start_site:
        source_bed = source_bed.strand_specific_start_site()

    # ---- build Bed(target) from atac.var (name = peak_name) ----
    atac_df = atac_var.copy()
    atac_df["name"] = atac_df[peak_name_col].astype(str)
    target_bed = Bed(atac_df)

    # ---- build edge list with node names ----
    ei = graph.edge_index
    if not torch.is_tensor(ei):
        ei = torch.as_tensor(ei, dtype=torch.long)
    else:
        ei = ei.long()
    ei = ei.detach().cpu().numpy()

    node_names = np.asarray(list(graph.node_names), dtype=object)
    edgelist = pd.DataFrame({"source": node_names[ei[0]], "target": node_names[ei[1]]})

    # ---- attach requested edge attributes ----
    if keep_attrs:
        if not hasattr(graph, "edge_attr_dict") or graph.edge_attr_dict is None:
            raise ValueError("`keep_attrs` is given but `graph.edge_attr_dict` is missing!")
        for k in keep_attrs:
            if k not in graph.edge_attr_dict:
                raise ValueError(f"Edge attribute `{k}` not found in `graph.edge_attr_dict`!")
            v = np.asarray(graph.edge_attr_dict[k])
            if v.shape[0] != edgelist.shape[0]:
                raise ValueError(f"Length mismatch for `{k}`: {v.shape[0]} != {edgelist.shape[0]}")
            edgelist[k] = v

    # ---- merge coordinates using Bed.name ----
    out = edgelist.merge(
        source_bed.df.iloc[:, :4], how="left", left_on="source", right_on="name"
    ).merge(
        target_bed.df.iloc[:, :4],
        how="left",
        left_on="target",
        right_on="name",
        suffixes=("_src", "_tgt"),
    )

    required = [
        "chrom_src",
        "chromStart_src",
        "chromEnd_src",
        "chrom_tgt",
        "chromStart_tgt",
        "chromEnd_tgt",
    ]
    missing_cols = [c for c in required if c not in out.columns]
    if missing_cols:
        raise ValueError(
            f"Unexpected merged columns. Missing: {missing_cols}. "
            f"Available: {list(out.columns)}"
        )

    miss = out[required].isna().any(axis=1)
    if miss.any():
        examples = out.loc[miss, ["source", "target"]].head(10)
        raise ValueError(
            "Some source/target names are not found in provided rna_var / atac_var name columns. Examples:\n"
            f"{examples.to_string(index=False)}"
        )

    out.loc[
        :,
        [
            "chrom_src",
            "chromStart_src",
            "chromEnd_src",
            "chrom_tgt",
            "chromStart_tgt",
            "chromEnd_tgt",
            *(keep_attrs or []),
        ],
    ].to_csv(file, sep="\t", index=False, header=False)


def cis_regulatory_ranking(
    gene2region: Data,
    region2tf: Data,
    genes: list[str],
    regions: list[str],
    tfs: list[str],
    region_lens: list[int] | None = None,
    n_samples: int = 1000,
    weight_attr: str | None = None,
    random_state: RandomState = None,
) -> pd.DataFrame:
    def _biadjacency(data, row_names, col_names, weight_attr=None):
        row_map = {name: i for i, name in enumerate(row_names)}
        col_map = {name: i for i, name in enumerate(col_names)}
        src = data.edge_index[0].tolist()
        dst = data.edge_index[1].tolist()

        weights_arr = None
        if weight_attr is not None:
            ead = getattr(data, "edge_attr_dict", None)
            if ead is not None and weight_attr in ead:
                weights_arr = np.asarray(ead[weight_attr], dtype=np.float32)

        rows, cols, vals = [], [], []
        for e, (s, d) in enumerate(zip(src, dst)):
            sn, dn = data.node_names[s], data.node_names[d]
            if sn in row_map and dn in col_map:
                rows.append(row_map[sn])
                cols.append(col_map[dn])
            elif dn in row_map and sn in col_map:
                rows.append(row_map[dn])
                cols.append(col_map[sn])
            else:
                continue
            w = float(weights_arr[e]) if weights_arr is not None else 1.0
            vals.append(w)

        dtype = np.float32 if weights_arr is not None else np.int16
        return scipy.sparse.csr_matrix(
            (np.array(vals, dtype=dtype), (rows, cols)),
            shape=(len(row_names), len(col_names)),
        )

    gene2region = _biadjacency(gene2region, genes, regions)
    region2tf = _biadjacency(region2tf, regions, tfs, weight_attr=weight_attr)

    if n_samples:
        region_lens = [1] * len(regions) if region_lens is None else region_lens
        if len(region_lens) != len(regions):
            raise ValueError("`region_lens` must have the same length as `regions`!")
        region_bins = pd.qcut(region_lens, min(len(set(region_lens)), 500), duplicates="drop")
        region_bins_lut = pd.RangeIndex(region_bins.size).groupby(region_bins)
        rs = get_rs(random_state)
        row, col_rand, data = [], [], []
        lil = gene2region.tolil()
        for r, (c, d) in tqdm(
            enumerate(zip(lil.rows, lil.data)),
            total=len(lil.rows),
            desc="cis_reg_ranking.sampling",
        ):
            if not c:
                continue
            row.append(np.ones_like(c) * r)
            col_rand.append(
                np.stack(
                    [
                        rs.choice(region_bins_lut[region_bins[c_]], n_samples, replace=True)
                        for c_ in c
                    ],
                    axis=0,
                )
            )
            data.append(d)
        if not row:
            raise ValueError(
                "gene2region matrix is empty — no gene–peak edges found. "
                "Possible causes: (1) g_sig has 0 edges (check Step 3 filtering output); "
                "(2) node names in g_sig don't match `genes`/`regions`. "
                f"g_sig edges: {gene2region.nnz}, genes matched: {gene2region.sum():.0f}"
            )
        row = np.concatenate(row)
        col_rand = np.concatenate(col_rand)
        data = np.concatenate(data)
        gene2tf_obs = (gene2region @ region2tf).toarray()
        _rand_dtype = np.float32 if weight_attr is not None else np.int16
        gene2tf_rand = np.empty((len(genes), len(tfs), n_samples), dtype=_rand_dtype)
        for k in tqdm(range(n_samples), desc="cis_reg_ranking.mapping"):
            gene2region_rand = scipy.sparse.coo_matrix(
                (data, (row, col_rand[:, k])), shape=(len(genes), len(regions))
            )
            gene2tf_rand[:, :, k] = (gene2region_rand @ region2tf).toarray()
        gene2tf_rand.sort(axis=2)
        gene2tf_enrich = np.empty_like(gene2tf_obs)
        for i, j in product(range(len(genes)), range(len(tfs))):
            if gene2tf_obs[i, j] == 0:
                gene2tf_enrich[i, j] = 0
                continue
            gene2tf_enrich[i, j] = np.searchsorted(
                gene2tf_rand[i, j, :], gene2tf_obs[i, j], side="right"
            )
    else:
        gene2tf_enrich = (gene2region @ region2tf).toarray()

    return pd.DataFrame(scipy.stats.rankdata(-gene2tf_enrich, axis=0), index=genes, columns=tfs)
