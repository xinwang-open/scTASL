"""
Data preprocessing and heterogeneous graph construction for paired scRNA-seq/scATAC-seq.

Input:
    - RNA h5ad file
    - ATAC h5ad file
Assumption:
    - paired data, with overlapping obs_names/cell barcodes
Output:
    - PyG HeteroData with three node types: cell, gene, peak
    - cell features: concat(RNA-PCA, ATAC-LSI)
    - gene features: PCA on gene-by-cell RNA profile
    - peak features: LSI on peak-by-cell ATAC profile
    - edges: cell-gene expression edges and cell-peak accessibility edges
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer


def _as_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def _as_csr(x):
    return x.tocsr() if sparse.issparse(x) else sparse.csr_matrix(x)


def load_and_align(rna_path: str, atac_path: str):
    """Load RNA/ATAC h5ad files and align cells by common obs_names."""
    print(f"[Load] RNA:  {rna_path}")
    rna = sc.read_h5ad(rna_path)
    print(f"[Load] ATAC: {atac_path}")
    atac = sc.read_h5ad(atac_path)

    common_cells = sorted(set(rna.obs_names) & set(atac.obs_names))
    print(f"[Align] RNA cells={rna.n_obs}, ATAC cells={atac.n_obs}, common={len(common_cells)}")
    if len(common_cells) == 0:
        raise ValueError("RNA and ATAC h5ad files have no overlapping obs_names/cell barcodes.")

    rna = rna[common_cells].copy()
    atac = atac[common_cells].copy()
    return rna, atac


def preprocess_rna(rna, n_hvg: int = 3000, min_cells: int = 10):
    """Standard RNA preprocessing: gene filtering, normalization, log1p, HVG selection."""
    print("[RNA] Preprocessing...")
    sc.pp.filter_genes(rna, min_cells=min_cells)
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=n_hvg, flavor="seurat")
    rna = rna[:, rna.var.highly_variable].copy()
    print(f"[RNA] After HVG selection: {rna.shape}")
    return rna


def preprocess_atac(atac, min_cells_frac: float = 0.01, max_cells_frac: float = 0.95):
    """Peak frequency filtering for ATAC matrix."""
    print("[ATAC] Preprocessing...")
    n_cells = atac.n_obs
    x = atac.X
    if sparse.issparse(x):
        peak_freq = np.asarray((x > 0).sum(axis=0)).ravel() / n_cells
    else:
        peak_freq = (np.asarray(x) > 0).sum(axis=0) / n_cells

    peak_mask = (peak_freq > min_cells_frac) & (peak_freq < max_cells_frac)
    atac = atac[:, peak_mask].copy()
    print(f"[ATAC] After peak frequency filtering: {atac.shape}")
    return atac


def compute_rna_cell_features(rna, n_comp: int = 50):
    """Cell features from RNA: PCA on log-normalized HVG expression."""
    print(f"[Feat] RNA cell features by PCA, dim={n_comp}")
    x = _as_dense(rna.X)
    n_comp_actual = min(n_comp, min(x.shape) - 1)
    if n_comp_actual <= 0:
        raise ValueError("RNA matrix is too small for PCA.")
    pca = PCA(n_components=n_comp_actual, random_state=42)
    feat = pca.fit_transform(x)
    if n_comp_actual < n_comp:
        feat = np.concatenate([feat, np.zeros((feat.shape[0], n_comp - n_comp_actual))], axis=1)
    print(
        f"[Feat] RNA cell features: {feat.shape}; explained variance={pca.explained_variance_ratio_.sum():.3f}"
    )
    return feat.astype(np.float32)


def compute_atac_cell_features(atac, n_comp: int = 50, drop_first: bool = True):
    """Cell features from ATAC: TF-IDF + SVD/LSI on cell-by-peak matrix."""
    print(f"[Feat] ATAC cell features by LSI, dim={n_comp}, drop_first={drop_first}")
    x = _as_csr(atac.X)
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
    x_tfidf = tfidf.fit_transform(x)

    # Need one extra component if the first LSI component is dropped.
    n_requested = n_comp + 1 if drop_first else n_comp
    n_comp_actual = min(n_requested, min(x_tfidf.shape) - 1)
    if n_comp_actual <= 0:
        raise ValueError("ATAC matrix is too small for LSI.")
    svd = TruncatedSVD(n_components=n_comp_actual, random_state=42)
    feat = svd.fit_transform(x_tfidf)
    if drop_first and feat.shape[1] > 1:
        feat = feat[:, 1:]

    # z-score each component
    feat = (feat - feat.mean(axis=0, keepdims=True)) / (feat.std(axis=0, keepdims=True) + 1e-8)
    if feat.shape[1] < n_comp:
        feat = np.concatenate([feat, np.zeros((feat.shape[0], n_comp - feat.shape[1]))], axis=1)
    feat = feat[:, :n_comp]
    print(f"[Feat] ATAC cell features: {feat.shape}")
    return feat.astype(np.float32)


def compute_gene_features(rna, n_comp: int = 50):
    """Gene features: PCA on gene-by-cell log-normalized RNA profile."""
    print(f"[Feat] Gene features by PCA on transposed RNA matrix, dim={n_comp}")
    x = _as_dense(rna.X).T  # gene x cell
    n_comp_actual = min(n_comp, min(x.shape) - 1)
    if n_comp_actual <= 0:
        raise ValueError("RNA matrix is too small for gene feature PCA.")
    pca = PCA(n_components=n_comp_actual, random_state=42)
    feat = pca.fit_transform(x)
    feat = (feat - feat.mean(axis=0, keepdims=True)) / (feat.std(axis=0, keepdims=True) + 1e-8)
    if n_comp_actual < n_comp:
        feat = np.concatenate([feat, np.zeros((feat.shape[0], n_comp - n_comp_actual))], axis=1)
    print(f"[Feat] Gene features: {feat.shape}")
    return feat.astype(np.float32)


def compute_peak_features(atac, n_comp: int = 50):
    """Peak features: TF-IDF + SVD/LSI on peak-by-cell ATAC profile."""
    print(f"[Feat] Peak features by LSI on transposed ATAC matrix, dim={n_comp}")
    x = _as_csr(atac.X).T  # peak x cell
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    x_tfidf = tfidf.fit_transform(x)
    n_comp_actual = min(n_comp, min(x_tfidf.shape) - 1)
    if n_comp_actual <= 0:
        raise ValueError("ATAC matrix is too small for peak feature LSI.")
    svd = TruncatedSVD(n_components=n_comp_actual, random_state=42)
    feat = svd.fit_transform(x_tfidf)
    feat = (feat - feat.mean(axis=0, keepdims=True)) / (feat.std(axis=0, keepdims=True) + 1e-8)
    if n_comp_actual < n_comp:
        feat = np.concatenate([feat, np.zeros((feat.shape[0], n_comp - n_comp_actual))], axis=1)
    print(f"[Feat] Peak features: {feat.shape}")
    return feat.astype(np.float32)


def compute_atac_tfidf_for_edges(atac):
    """Return cell-by-peak TF-IDF matrix for building weighted cell-peak edges."""
    x = _as_csr(atac.X)
    tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
    return tfidf.fit_transform(x).tocsr()


def build_edges_topk(matrix, k: int, mode: str = "gene"):
    """
    Build top-k cell-feature edges row by row.

    Args:
        matrix: cell-by-feature matrix. Sparse matrices are processed without densifying.
        k: maximum number of feature neighbors per cell.
        mode: only used for logging.

    Returns:
        edge_index: shape [2, n_edges], source cell index and target feature index.
        edge_weight: shape [n_edges].
    """
    print(f"[Edge] Building top-{k} cell-{mode} edges...")
    if k <= 0:
        raise ValueError("k must be positive.")

    cell_indices = []
    feat_indices = []
    weights = []

    if sparse.issparse(matrix):
        x = matrix.tocsr()
        n_cell, n_feat = x.shape
        for i in range(n_cell):
            start, end = x.indptr[i], x.indptr[i + 1]
            inds = x.indices[start:end]
            vals = x.data[start:end]
            mask = vals > 0
            inds = inds[mask]
            vals = vals[mask]
            if vals.size == 0:
                continue
            if vals.size > k:
                top = np.argpartition(-vals, k - 1)[:k]
                inds = inds[top]
                vals = vals[top]
            cell_indices.append(np.full(inds.shape, i, dtype=np.int64))
            feat_indices.append(inds.astype(np.int64))
            weights.append(vals.astype(np.float32))
    else:
        x = np.asarray(matrix)
        n_cell, n_feat = x.shape
        k_actual = min(k, n_feat)
        topk_idx = np.argpartition(-x, k_actual - 1, axis=1)[:, :k_actual]
        row_idx = np.repeat(np.arange(n_cell), k_actual)
        col_idx = topk_idx.reshape(-1)
        vals = x[row_idx, col_idx]
        mask = vals > 0
        cell_indices.append(row_idx[mask].astype(np.int64))
        feat_indices.append(col_idx[mask].astype(np.int64))
        weights.append(vals[mask].astype(np.float32))

    if len(cell_indices) == 0:
        raise ValueError(f"No positive cell-{mode} edges were built. Check the input matrix.")

    cell_idx = np.concatenate(cell_indices)
    feat_idx = np.concatenate(feat_indices)
    edge_weight = np.concatenate(weights).astype(np.float32)

    # Keep edge_attr non-negative and scale-stable. HGTConv below does not consume edge_attr,
    # but saving it is useful for future weighted extensions.
    edge_weight = edge_weight / (edge_weight.mean() + 1e-8)

    edge_index = np.stack([cell_idx, feat_idx], axis=0).astype(np.int64)
    print(
        f"[Edge] cell-{mode}: {edge_index.shape[1]} edges; avg degree={edge_index.shape[1] / n_cell:.1f}"
    )
    return edge_index, edge_weight


def build_hetero_data(
    rna,
    atac,
    rna_cell_feat: np.ndarray,
    atac_cell_feat: np.ndarray,
    gene_feat: np.ndarray,
    peak_feat: np.ndarray,
    k_gene: int = 500,
    k_peak: int = 1000,
):
    """Build a PyG HeteroData object."""
    from torch_geometric.data import HeteroData

    print("[Graph] Building cell-gene-peak heterogeneous graph...")
    data = HeteroData()

    # Cell feature is a concatenation, but train.py will mask it into RNA-only,
    # ATAC-only, or full features depending on graph view.
    cell_feat = np.concatenate([rna_cell_feat, atac_cell_feat], axis=1).astype(np.float32)
    data["cell"].x = torch.from_numpy(cell_feat)
    data["gene"].x = torch.from_numpy(gene_feat)
    data["peak"].x = torch.from_numpy(peak_feat)

    data["cell"].rna_dim = int(rna_cell_feat.shape[1])
    data["cell"].atac_dim = int(atac_cell_feat.shape[1])

    # Cell-gene edges from log-normalized RNA expression.
    rna_x = rna.X.tocsr() if sparse.issparse(rna.X) else sparse.csr_matrix(rna.X)
    edge_cg, weight_cg = build_edges_topk(rna_x, k=k_gene, mode="gene")
    data["cell", "expresses", "gene"].edge_index = torch.from_numpy(edge_cg)
    data["cell", "expresses", "gene"].edge_attr = torch.from_numpy(weight_cg).unsqueeze(-1)
    data["gene", "rev_expresses", "cell"].edge_index = torch.from_numpy(edge_cg[[1, 0]])
    data["gene", "rev_expresses", "cell"].edge_attr = torch.from_numpy(weight_cg).unsqueeze(-1)

    # Cell-peak edges from TF-IDF weighted ATAC accessibility, not raw binary counts.
    atac_tfidf = compute_atac_tfidf_for_edges(atac)
    edge_cp, weight_cp = build_edges_topk(atac_tfidf, k=k_peak, mode="peak")
    data["cell", "accesses", "peak"].edge_index = torch.from_numpy(edge_cp)
    data["cell", "accesses", "peak"].edge_attr = torch.from_numpy(weight_cp).unsqueeze(-1)
    data["peak", "rev_accesses", "cell"].edge_index = torch.from_numpy(edge_cp[[1, 0]])
    data["peak", "rev_accesses", "cell"].edge_attr = torch.from_numpy(weight_cp).unsqueeze(-1)

    print(
        f"[Graph] Nodes: cell={data['cell'].num_nodes}, gene={data['gene'].num_nodes}, peak={data['peak'].num_nodes}"
    )
    print(
        f"[Graph] Edges: cell-gene={data['cell', 'expresses', 'gene'].edge_index.shape[1]}, "
        f"cell-peak={data['cell', 'accesses', 'peak'].edge_index.shape[1]}"
    )
    print(
        f"[Graph] Cell feature dim: RNA={rna_cell_feat.shape[1]}, ATAC={atac_cell_feat.shape[1]}, total={cell_feat.shape[1]}"
    )
    return data


def prepare_data(
    rna_path: str,
    atac_path: str,
    n_hvg: int = 3000,
    n_comp: int = 50,
    k_gene: int = 500,
    k_peak: int = 1000,
    min_cells: int = 10,
    min_peak_frac: float = 0.01,
    max_peak_frac: float = 0.95,
):
    """End-to-end data preparation pipeline."""
    rna, atac = load_and_align(rna_path, atac_path)
    rna = preprocess_rna(rna, n_hvg=n_hvg, min_cells=min_cells)
    atac = preprocess_atac(atac, min_cells_frac=min_peak_frac, max_cells_frac=max_peak_frac)

    rna_cell_feat = compute_rna_cell_features(rna, n_comp=n_comp)
    atac_cell_feat = compute_atac_cell_features(atac, n_comp=n_comp, drop_first=True)
    gene_feat = compute_gene_features(rna, n_comp=n_comp)
    peak_feat = compute_peak_features(atac, n_comp=n_comp)

    data = build_hetero_data(
        rna=rna,
        atac=atac,
        rna_cell_feat=rna_cell_feat,
        atac_cell_feat=atac_cell_feat,
        gene_feat=gene_feat,
        peak_feat=peak_feat,
        k_gene=k_gene,
        k_peak=k_peak,
    )
    return data, rna, atac
