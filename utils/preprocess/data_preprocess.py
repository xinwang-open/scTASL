import re
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import pybedtools
import torch
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from tqdm import tqdm


def add_peak_coordinates(
    atac_adata: ad.AnnData,
    peak_key: str | None = None,
    inplace: bool = True,
    zero_based: bool = True,
) -> ad.AnnData | None:
    """
    Parse peak identifiers and populate ``chrom``, ``chromStart`` and ``chromEnd`` in ``atac_adata.var``.

    Parameters
    ----------
    atac_adata
        ATAC AnnData object containing peak information.
    peak_key
        Column in ``atac_adata.var`` holding peak strings like ``chr1:100-200``.
        If ``None`` (default), use ``atac_adata.var_names``.
    inplace
        Whether to modify ``atac_adata`` in place. If False, return a new AnnData copy.
    zero_based
        If True (default), subtract 1 from the parsed ``chromStart`` to convert to 0-based indexing.

    Returns
    -------
    Optional[AnnData]
        None if operating in place, otherwise the modified AnnData copy.
    """
    target = atac_adata if inplace else atac_adata.copy()

    if peak_key is None:
        peak_series = target.var_names.to_series().astype(str)
    else:
        if peak_key not in target.var.columns:
            raise KeyError(f"Column '{peak_key}' not found in atac_adata.var.")
        peak_series = target.var[peak_key].astype(str)

    parts = peak_series.str.split(":", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("Peak identifiers must follow the 'chrom:start-end' format.")
    chrom = parts[0]
    pos = parts[1].str.split("-", n=1, expand=True)
    if pos.shape[1] != 2:
        raise ValueError("Peak identifiers must follow the 'chrom:start-end' format.")

    chrom_start = pd.to_numeric(pos[0], errors="coerce")
    chrom_end = pd.to_numeric(pos[1], errors="coerce")
    if chrom_start.isna().any() or chrom_end.isna().any():
        raise ValueError("Peak identifiers contain non-numeric start or end positions.")

    chrom_start = chrom_start.astype("Int64")
    chrom_end = chrom_end.astype("Int64")
    if zero_based:
        chrom_start = chrom_start - 1

    peak_df = pd.DataFrame(
        {
            "chrom": chrom.astype(str).values,
            "chromStart": chrom_start,
            "chromEnd": chrom_end,
        },
        index=target.var.index,
    )

    target.var["peak_name"] = peak_series.values
    target.var[["chrom", "chromStart", "chromEnd"]] = peak_df

    return None if inplace else target


def annotate_rna_from_gtf(
    rna_adata: ad.AnnData,
    gtf_path: str,
    feature_type: str = "gene",
    inplace: bool = True,
    strip_chr_prefix: bool = False,
    input_id_type: Literal["auto", "symbol", "ensembl"] | None = "auto",
) -> ad.AnnData | None:
    """
    Populate RNA AnnData.var with genomic annotations from a GTF file.

    Parameters
    ----------
    rna_adata : AnnData
        RNA modality AnnData object.
    gtf_path : str
        Path to the GTF file.
    feature_type : str
        Feature type to retain (default: ``gene``).
    inplace : bool
        Modify ``rna_adata`` directly if True, otherwise return a copy.
    strip_chr_prefix : bool
        Remove ``chr`` prefix from chromosome names when True.
    input_id_type : {"auto", "symbol", "ensembl"}
        Hint for matching strategy. ``auto`` attempts to detect the identifier type.
    """
    cols = [
        "chrom",
        "source",
        "feature",
        "chromStart",
        "chromEnd",
        "score",
        "strand",
        "frame",
        "attribute",
    ]
    gtf_df = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        names=cols,
        dtype={"chrom": str, "chromStart": int, "chromEnd": int, "strand": str, "feature": str},
        na_values=".",
    )
    gtf_df = gtf_df[gtf_df["feature"] == feature_type].copy()
    if gtf_df.empty:
        raise ValueError(f"No entries of feature type '{feature_type}' found in {gtf_path}.")

    def _parse_attributes(attr: str) -> Mapping[str, str]:
        parsed = {}
        if isinstance(attr, str):
            for item in attr.strip().split(";"):
                item = item.strip()
                if not item or " " not in item:
                    continue
                key, value = item.split(" ", 1)
                parsed[key] = value.strip().strip('"')
        return parsed

    attr_df = gtf_df["attribute"].apply(_parse_attributes).apply(pd.Series)

    def _remove_version(x: str | None) -> str | None:
        if isinstance(x, str):
            return x.split(".", 1)[0]
        return x

    if "gene_id" in attr_df:
        attr_df["gene_id_novers"] = attr_df["gene_id"].map(_remove_version)

    gtf_df = pd.concat([gtf_df.drop(columns="attribute"), attr_df], axis=1)

    if strip_chr_prefix:
        gtf_df["chrom"] = gtf_df["chrom"].str.replace(r"^chr", "", regex=True)

    gtf_df["chromStart"] = gtf_df["chromStart"].astype(int)
    gtf_df["chromEnd"] = gtf_df["chromEnd"].astype(int)
    gtf_df["name"] = gtf_df["gene_name"] if "gene_name" in gtf_df.columns else gtf_df.get("gene_id")

    keep_cols = ["chrom", "chromStart", "chromEnd", "name", "score", "strand"]
    for extra in ["gene_id", "gene_id_novers", "gene_name", "gene_type", "level", "mgi_id"]:
        if extra in gtf_df.columns and extra not in keep_cols:
            keep_cols.append(extra)

    target = rna_adata if inplace else rna_adata.copy()

    def _looks_like_ensembl(values: pd.Series) -> bool:
        pattern = re.compile(r"^ENS[A-Z0-9]*G\\d+", re.IGNORECASE)
        return values.astype(str).str.match(pattern).mean() > 0.5

    def _guess_id_type() -> Literal["symbol", "ensembl"]:
        idx = target.var_names.astype(str)
        if _looks_like_ensembl(idx):
            return "ensembl"
        for key in ["gene_name", "symbol"]:
            if key in target.var.columns and not _looks_like_ensembl(target.var[key].astype(str)):
                return "symbol"
        return "symbol"

    id_type = input_id_type or "auto"
    if id_type == "auto":
        id_type = _guess_id_type()

    if "ensembl_id" not in target.var.columns:
        if _looks_like_ensembl(target.var_names.astype(str)):
            target.var["ensembl_id"] = target.var_names.astype(str)
        else:
            for candidate in ["gene_id", "gene_ids", "ensembl", "ensembl_gene_id"]:
                if candidate in target.var.columns:
                    target.var["ensembl_id"] = target.var[candidate].astype(str)
                    break

    if "ensembl_id" in target.var.columns and "ensembl_id_novers" not in target.var.columns:
        target.var["ensembl_id_novers"] = target.var["ensembl_id"].astype(str).map(_remove_version)

    if "gene_name" not in target.var.columns:
        for candidate in ["symbol", "gene_symbols", "gene", "hgnc_symbol"]:
            if candidate in target.var.columns:
                target.var["gene_name"] = target.var[candidate].astype(str)
                break
        else:
            if not _looks_like_ensembl(target.var_names.astype(str)):
                target.var["gene_name"] = target.var_names.astype(str)

    gtf_keys = [
        key for key in ["gene_id_novers", "gene_id", "gene_name", "name"] if key in gtf_df.columns
    ]

    var_keys: list[str] = []
    if id_type == "ensembl":
        for key in ["ensembl_id_novers", "ensembl_id", "gene_id"]:
            if key in target.var.columns:
                var_keys.append(key)
        for key in ["gene_name", "symbol"]:
            if key in target.var.columns and key not in var_keys:
                var_keys.append(key)
    else:
        for key in ["gene_name", "symbol"]:
            if key in target.var.columns:
                var_keys.append(key)
        for key in ["ensembl_id_novers", "ensembl_id", "gene_id"]:
            if key in target.var.columns and key not in var_keys:
                var_keys.append(key)

    target.var["_index_tmp_"] = target.var.index.astype(str)
    var_keys.append("_index_tmp_")

    merged = None
    for vkey in var_keys:
        if vkey not in target.var.columns:
            continue
        for gkey in gtf_keys:
            try:
                right_cols = [gkey] + [col for col in keep_cols if col != gkey]
                right = gtf_df[right_cols].dropna(subset=[gkey])
                if right.empty:
                    continue
                right = right.drop_duplicates(subset=[gkey], keep="first")

                left = target.var[[vkey]].copy()
                left["_orig_index_"] = left.index
                merged_tmp = left.merge(
                    right, how="left", left_on=vkey, right_on=gkey, suffixes=("", "_gtf")
                ).set_index("_orig_index_")

                annotation_cols = [col for col in keep_cols if col not in (vkey, gkey)]
                if merged_tmp[annotation_cols].notna().any().any():
                    merged = merged_tmp
                    break
            except Exception:
                continue
        if merged is not None:
            break

    if merged is None:
        raise RuntimeError("Failed to align AnnData genes with entries in the GTF file.")

    for col in keep_cols:
        if col in merged.columns:
            target.var[col] = merged[col].values
        elif f"{col}_gtf" in merged.columns:
            target.var[col] = merged[f"{col}_gtf"].values

    for col in ("chromStart", "chromEnd"):
        if col in target.var.columns:
            target.var[col] = pd.to_numeric(target.var[col], errors="coerce").astype("Int64")

    target.var.drop(columns=["_index_tmp_"], inplace=True, errors="ignore")

    return None if inplace else target


def expand_bed(
    bed_data: pd.DataFrame, upstream: int, downstream: int, chr_len: Mapping[str, int] | None = None
) -> pd.DataFrame:
    """
    Expand genomic features towards upstream and downstream

    Parameters
    ----------
    bed_data : pd.DataFrame
        The original DataFrame containing the genomic features.
        It must have columns 'chrom', 'chromStart', 'chromEnd', and 'strand'.
    upstream : int
        Number of bps to expand in the upstream direction
    downstream : int
        Number of bps to expand in the downstream direction
    chr_len : Optional[Mapping[str, int]]
        Length of each chromosome

    Returns
    -------
    expanded_bed : pd.DataFrame
        A new DataFrame containing expanded features

    Note
    ----
    Starting position < 0 after expansion is always trimmed.
    Ending position exceeding chromosome length is trimmed only if
    ``chr_len`` is specified.
    """
    if upstream == downstream == 0:
        return bed_data

    df = bed_data.copy()

    if upstream == downstream:
        df["chromStart"] -= upstream
        df["chromEnd"] += downstream
    else:  # asymmetric expansion
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

    # Trim chromStart if less than 0
    df["chromStart"] = np.maximum(df["chromStart"], 0)

    if chr_len:
        # Apply chromosome length trimming
        chr_len_series = df["chrom"].map(chr_len)
        df["chromEnd"] = np.minimum(df["chromEnd"], chr_len_series)

    return df


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


def window_matrices(
    left: pybedtools.BedTool | str,
    right: pybedtools.BedTool | str,
    window_size: int,
    left_sorted: bool = False,
    right_sorted: bool = False,
    attr_fn: Callable[[Interval, Interval, float], Mapping[str, Any]] | None = None,
) -> tuple:
    """
    Construct a window graph between two sets of genomic features, returning matrices instead of a graph.
    Features within a window size are connected.

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
        It should accept three arguments:
        - l: left interval
        - r: right interval
        - d: signed distance between the intervals.
        By default, no edge attribute is created.

    Returns
    -------
    edge_index_matrix
        Matrix indicating which nodes are connected (edges).
    edge_attr_matrix
        Matrix containing edge attributes like distance, weight, etc.
    """

    # Prepare left and right BedTool objects

    pbar_total = len(left)
    if not left_sorted:
        left = left.sort(stream=True)
    left = iter(left)

    if not right_sorted:
        right = right.sort(stream=True)
    right = iter(right)

    if pbar_total is not None:
        left = tqdm(left, total=pbar_total, desc="Processing TG-RE pairs")
    # Edge lists and attributes
    edges = []
    edges_attr = []
    window = deque()  # Using deque for ordered removal

    # Iterate through intervals and compute distances
    for l in left:
        for r in list(window):  # Allow remove during iteration
            d = interval_dist(l, r)
            if -window_size <= d <= window_size:
                edges.append((l.name, r.name))
                edges_attr.append(abs(d))  # Add edge attributes
            elif d > window_size:
                window.remove(r)  # Remove from window if distance exceeds
            else:  # dist < -window_size
                break  # No need to expand window
        else:
            for r in right:  # Resume from last break
                d = interval_dist(l, r)
                if -window_size <= d <= window_size:
                    edges.append((l.name, r.name))
                    edges_attr.append(abs(d))  # Add edge attributes
                elif d > window_size:
                    continue
                window.append(r)  # Add to window
                if d < -window_size:
                    break

    # Convert edge list to indices (row, col) format

    edges = np.array([list(item) for item in edges])
    edges_attr = np.array(edges_attr)
    assert len(edges) == len(edges_attr), "The number of edges and attributes must match"

    return edges, edges_attr


"""
Modified build_peak_to_gene_matrix function
Copy this to replace the original function in your data_preprocess.py
"""


def build_gene_peak_graph(
    rna_adata, atac_adata, upstream: int = 2000, downstream_len: int = 0, save_path=None
):
    """
    Build gene-peak regulatory matrix and convert to PyG graph object

    Parameters
    ----------
    rna_adata : AnnData
        RNA expression data
    atac_adata : AnnData
        ATAC peak data
    prompter_len : int
        Promoter region length (upstream of TSS)
    downstream_len : int
        Downstream extension length from TSS
    save_path : str, optional
        Path to save filtered ATAC data

    Returns
    -------
    filtered_atac : AnnData
        Filtered ATAC data (only peaks with connections)
    graph_data : torch_geometric.data.Data
        PyG graph object containing:
        - edge_index: [2, num_edges] edge indices
        - edge_attr: [num_edges, 1] edge attributes (distance)
        - num_nodes: total number of nodes
        - node_names: list of node names
        - node_type: node type (0=gene, 1=peak)
        - gene_names: list of gene names
        - peak_names: list of peak names
    """

    rna_df = rna_adata.var.reset_index()
    assert "strand" in rna_df.columns, "RNA data must contain strand information"

    # Expand RNA regions and create RNA BED
    rna_region_df = expand_bed(
        rna_df,
        upstream=upstream,
        downstream=downstream_len,
    )
    rna_region_df = rna_region_df[
        ["chrom", "chromStart", "chromEnd", "gene_name", "score", "strand"]
    ]
    rna_region_df["chromStart"] = rna_region_df["chromStart"].astype(int)
    rna_region_df["chromEnd"] = rna_region_df["chromEnd"].astype(int)
    rna_bed = BedTool.from_dataframe(rna_region_df)

    # Create BED format DataFrame for ATAC peaks
    atac_df = atac_adata.var.reset_index()
    atac_df["score"] = 0
    atac_df["strand"] = "."
    atac_df["chromStart"] = atac_df["chromStart"].astype(int)
    atac_df["chromEnd"] = atac_df["chromEnd"].astype(int)
    atac_df = atac_df[["chrom", "chromStart", "chromEnd", "peak_name", "score", "strand"]]
    atac_bed = BedTool.from_dataframe(atac_df)

    # Calculate peak-to-gene connections
    edges, edges_attr = window_matrices(
        left=rna_bed,
        right=atac_bed,
        window_size=0,
    )

    # Create gene-to-peak index mapping
    gene_names = rna_df["gene_name"].tolist()
    peak_names = atac_df["peak_name"].tolist()
    gene_idx = {g: i for i, g in enumerate(gene_names)}
    peak_idx = {p: i for i, p in enumerate(peak_names)}

    # Create sparse matrix
    rows, cols = zip(*[(gene_idx[e[0]], peak_idx[e[1]]) for e in edges])
    data_values = [1] * len(rows)
    gene_peak_matrix_csr = csr_matrix(
        (data_values, (rows, cols)), shape=(len(gene_names), len(peak_names))
    )
    gene_peak_attr_matrix = csr_matrix(
        (edges_attr, (rows, cols)), shape=(len(gene_names), len(peak_names))
    )

    # Filter peaks that have connections
    peak_connections = np.array(gene_peak_matrix_csr.sum(axis=0)).flatten()
    connected_peak_mask = peak_connections > 0
    filtered_atac = atac_adata[:, connected_peak_mask].copy()

    print(f"original_atac_shape: {atac_adata.shape}")
    print(f"filtered_atac_shape: {filtered_atac.shape}")

    if save_path:
        filtered_atac.write(save_path)

    # Filter matrices
    filtered_gene_peak_matrix = gene_peak_matrix_csr[:, connected_peak_mask]
    filtered_edge_attr_matrix = gene_peak_attr_matrix[:, connected_peak_mask]

    # Get filtered peak names
    filtered_peak_names = [peak_names[i] for i in range(len(peak_names)) if connected_peak_mask[i]]

    # ============================================================
    # Convert to PyG graph object
    # ============================================================
    n_genes = len(gene_names)
    n_peaks = len(filtered_peak_names)

    # 1. Build node name list
    node_names = gene_names + filtered_peak_names
    num_nodes = len(node_names)

    # 2. Extract edges (COO format)
    adj_coo = filtered_gene_peak_matrix.tocoo()

    # Peak node indices need offset
    peak_offset = n_genes
    edge_index = np.vstack(
        [adj_coo.row, adj_coo.col + peak_offset]  # gene indices  # peak indices (with offset)
    )
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # 3. Extract edge attributes (distance)
    attr_values = []
    for i in range(len(adj_coo.row)):
        g_idx, p_idx = adj_coo.row[i], adj_coo.col[i]
        attr_values.append(filtered_edge_attr_matrix[g_idx, p_idx])
    edge_attr = torch.tensor(attr_values, dtype=torch.float).unsqueeze(1)

    # 4. Node type labels
    node_type = torch.cat(
        [
            torch.zeros(n_genes, dtype=torch.long),  # 0: gene
            torch.ones(n_peaks, dtype=torch.long),  # 1: peak
        ]
    )

    # 5. Create PyG Data object
    graph_data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

    # Add additional information
    graph_data.node_names = node_names
    graph_data.node_type = node_type
    graph_data.gene_names = gene_names
    graph_data.peak_names = filtered_peak_names

    print("PyG graph created:")
    print(f"  - num_nodes: {graph_data.num_nodes}")
    print(f"  - num_edges: {graph_data.num_edges}")
    print(f"  - edge_attr shape: {graph_data.edge_attr.shape}")

    return filtered_atac, graph_data
