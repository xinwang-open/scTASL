r"""
Performance evaluation metrics
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.spatial
import sklearn.metrics
import sklearn.neighbors
from anndata import AnnData
from scipy.sparse.csgraph import connected_components
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize



def normalized_mutual_information(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Normalized mutual information with true clustering
    """
    x = AnnData(X=x)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    nmi_list = []
    for res in (np.arange(20) + 1) / 10:
        sc.tl.leiden(x, resolution=res, n_iterations=2)
        leiden = x.obs["leiden"]
        score = sklearn.metrics.normalized_mutual_info_score(y, leiden, **kwargs)
        # 关键就这一行：兼容 float / np.float64
        nmi_list.append(np.asarray(score))
    return max(nmi_list)


def adjusted_rand_index(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Adjusted_rand_index with true clustering

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.normalized_mutual_info_score`

    Returns
    -------
    nmi
        Normalized mutual information

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    x = AnnData(X=x)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    nmi_list = []
    for res in (np.arange(20) + 1) / 10:
        sc.tl.leiden(x, resolution=res, n_iterations=2)
        leiden = x.obs["leiden"]
        nmi_list.append(sklearn.metrics.adjusted_rand_score(y, leiden, **kwargs))
    return max(nmi_list)


"""-----------------------------------kBET STAR--------------------------------------------"""


def kBET(data, batch_labels, k=50, alpha=0.05):
    """
    Python implementation of kBET with mean chi-square statistic (stat_mean).

    Parameters:
    - data: ndarray, shape (n_samples, n_features)
        The embedded data (e.g., PCA or UMAP embeddings).
    - batch_labels: ndarray, shape (n_samples,)
        Batch labels for each sample.
    - k: int, default=50
        Number of neighbors for k-NN.
    - alpha: float, default=0.05
        Significance threshold for p-value.

    Returns:
    - acceptance_rate: float
        The kBET acceptance rate (proportion of non-rejected samples).
    - stat_mean: float
        Mean kBET chi-square statistic over all cells.
    - p_values: ndarray
        Array of p-values for each sample.
    """
    # Ensure batch_labels are integers for bincount
    unique_batches = np.unique(batch_labels)
    batch_mapping = {label: i for i, label in enumerate(unique_batches)}
    batch_labels = np.array([batch_mapping[label] for label in batch_labels])

    # Step 1: Compute k-NN
    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 to include the sample itself
    nn.fit(data)
    neighbors = nn.kneighbors(data, return_distance=False)[:, 1:]  # Exclude itself

    # Step 2: Perform kBET for each sample
    p_values = []
    chi_square_stats = []
    for i, neigh in enumerate(neighbors):
        # Observed batch counts in the neighborhood
        observed_counts = np.bincount(batch_labels[neigh], minlength=len(unique_batches))
        # Expected uniform distribution
        expected_counts = np.full_like(observed_counts, k / len(unique_batches))

        # Chi-squared test
        chi2_stat, p_value, _, _ = chi2_contingency([observed_counts, expected_counts])
        p_values.append(p_value)
        chi_square_stats.append(chi2_stat)

    # Step 3: Compute acceptance rate and mean chi-square statistic
    p_values = np.array(p_values)
    chi_square_stats = np.array(chi_square_stats)
    acceptance_rate = np.mean(p_values > alpha)
    stat_mean = np.mean(chi_square_stats)

    return acceptance_rate, stat_mean, p_values


"""-----------------------------------kBET END--------------------------------------------"""


"""-------------------------------Batch Entropy START------------------------------------"""


def compute_omics_entropy(data, omics_labels, k=15):
    """
    Compute Omics Entropy for batch effect evaluation.

    Parameters:
    - data: ndarray, shape (n_samples, n_features)
        Embedded data (e.g., PCA or UMAP embeddings).
    - omics_labels: ndarray, shape (n_samples,)
        Omics/batch labels for each sample.
    - k: int, default=15
        Number of neighbors for k-NN.

    Returns:
    - omics_entropies: ndarray, shape (n_samples,)
        Entropy values for each sample.
    - mean_omics_entropy: float
        Mean omics entropy across all samples.
    """
    # Ensure omics_labels are integers for bincount
    unique_omics = np.unique(omics_labels)
    omics_mapping = {label: i for i, label in enumerate(unique_omics)}
    omics_labels = np.array([omics_mapping[label] for label in omics_labels])

    # Step 1: Compute k-NN
    nn = NearestNeighbors(n_neighbors=k + 1)  # Include the sample itself
    nn.fit(data)
    neighbors = nn.kneighbors(data, return_distance=False)[:, 1:]  # Exclude itself

    # Step 2: Compute Omics Entropy for each sample
    def entropy(probabilities):
        """Compute entropy given a probability distribution."""
        return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add epsilon to avoid log(0)

    omics_entropies = []
    for i, neigh in enumerate(neighbors):
        # Observed omics-label counts in the neighborhood
        counts = np.bincount(omics_labels[neigh], minlength=len(unique_omics))
        probabilities = counts / counts.sum()

        # Compute entropy
        omics_entropy = entropy(probabilities)
        omics_entropies.append(omics_entropy)

    omics_entropies = np.array(omics_entropies)

    # Step 3: Compute mean omics entropy
    mean_omics_entropy = np.mean(omics_entropies)

    return omics_entropies, mean_omics_entropy


"""-------------------------------Batch Entropy END------------------------------------"""


"""--------------------Biological Conservation Score (BCS) START-----------------------"""


def compute_bcs(data, cell_labels, test_size=0.2, random_state=42):
    """
    Compute Biological Conservation Score (BCS).

    Parameters:
    - data: ndarray, shape (n_samples, n_features)
        Embedded data (e.g., PCA or UMAP embeddings).
    - cell_labels: ndarray, shape (n_samples,)
        Cell type labels for each sample.
    - test_size: float, default=0.2
        Proportion of data to use as the test set.
    - random_state: int, default=42
        Random seed for reproducibility.

    Returns:
    - bcs_score: float
        Biological Conservation Score (BCS), based on classification accuracy.
    """
    # Step 1: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, cell_labels, test_size=test_size, random_state=random_state
    )

    # Step 2: Train a classifier (e.g., Random Forest)
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    # Step 3: Evaluate classification accuracy
    y_pred = clf.predict(X_test)
    bcs_score = accuracy_score(y_test, y_pred)

    return bcs_score


"""--------------------Biological Conservation Score (BCS) END-----------------------"""


def avg_silhouette_width_label(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    """
    Cell type average silhouette width (macro over cell types)
    - Each cell type contributes equally (avoid big classes dominating)
    """
    # 每个细胞的 silhouette（范围 [-1, 1]）
    s = sklearn.metrics.silhouette_samples(x, y, **kwargs)

    # 每个 cell type 先取均值，再对所有 cell type 平均（macro）
    df = pd.DataFrame({"y": y, "s": s})
    score = df.groupby("y", observed=True)["s"].mean().mean()

    # 映射到 [0, 1]
    return (float(score) + 1.0) / 2.0


def graph_connectivity(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Graph connectivity

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`scanpy.pp.neighbors`

    Returns
    -------
    conn
        Graph connectivity
    """
    x = AnnData(X=x)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X", **kwargs)
    conns = []
    for y_ in np.unique(y):
        x_ = x[y == y_]
        _, c = connected_components(x_.obsp["connectivities"], connection="strong")
        # counts = pd.value_counts(c)
        counts = pd.Series(c).value_counts()
        conns.append(counts.max() / counts.sum())
    return np.mean(conns)


def avg_silhouette_width_batch(x: np.ndarray, y: np.ndarray, ct: np.ndarray, **kwargs) -> float:
    r"""
    Batch average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    ct
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.metrics.silhouette_samples`

    Returns
    -------
    asw_batch
        Batch average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    s_per_ct = []
    for t in np.unique(ct):
        mask = ct == t
        try:
            s = sklearn.metrics.silhouette_samples(x[mask], y[mask], **kwargs)
        except ValueError:  # Too few samples
            s = 0
        s = (1 - np.fabs(s)).mean()
        s_per_ct.append(s)
    return np.mean(s_per_ct)


def foscttm(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    r"""
    Fraction of samples closer than true match (smaller is better)

    Parameters
    ----------
    x
        Coordinates for samples in modality X
    y
        Coordinates for samples in modality y
    **kwargs
        Additional keyword arguments are passed to
        :func:`scipy.spatial.distance_matrix`

    Returns
    -------
    float
        Mean FOSCTTM score averaged over both modalities (smaller is better)

    Note
    ----
    Samples in modality X and Y should be paired and given in the same order
    """
    if x.shape != y.shape:
        raise ValueError("Shapes do not match!")
    d = scipy.spatial.distance_matrix(x, y, **kwargs)
    foscttm_x = (d < np.expand_dims(np.diag(d), axis=1)).mean(axis=1)
    foscttm_y = (d < np.expand_dims(np.diag(d), axis=0)).mean(axis=0)
    return 0.5 * (foscttm_x.mean() + foscttm_y.mean())


"""--------------------Transfer accuracy start-----------------------"""


def _prepare_labels(
    labels_ref: Sequence[str],
    labels_query: Sequence[str],
    restrict_to_intersection: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align label spaces between reference and query.
    If restrict_to_intersection=True, only evaluate query cells whose labels
    exist in the reference set (common practice for transfer accuracy).
    Returns:
        labels_ref_np: np.ndarray of reference labels
        labels_query_filtered: np.ndarray of filtered query labels
        mask_query: boolean mask selecting evaluated query cells
    """
    labels_ref_np = np.asarray(labels_ref)
    labels_query_np = np.asarray(labels_query)

    if restrict_to_intersection:
        ref_set = set(labels_ref_np)
        mask_query = np.array([lab in ref_set for lab in labels_query_np], dtype=bool)
        labels_query_filtered = labels_query_np[mask_query]
    else:
        mask_query = np.ones(len(labels_query_np), dtype=bool)
        labels_query_filtered = labels_query_np

    return labels_ref_np, labels_query_filtered, mask_query


def transfer_accuracy_embeddings(
    X_ref: np.ndarray,
    y_ref: Sequence[str],
    X_query: np.ndarray,
    y_query: Sequence[str],
    k: int = 5,
    metric: str = "cosine",  # "cosine" or "euclidean" are most common
    l2_normalize: bool = True,
    restrict_to_intersection: bool = True,
    weights: str = "uniform",  # "uniform" or "distance"
) -> float:
    """
    One-way kNN label transfer: train on (X_ref, y_ref), predict labels for X_query,
    compare to y_query, return accuracy.
    """
    X_ref = np.asarray(X_ref)
    X_query = np.asarray(X_query)

    if l2_normalize:
        X_ref = normalize(X_ref, norm="l2", axis=1)
        X_query = normalize(X_query, norm="l2", axis=1)

    y_ref_np, y_query_eval, mask_q = _prepare_labels(y_ref, y_query, restrict_to_intersection)
    if y_query_eval.size == 0:
        return float("nan")

    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    knn.fit(X_ref, y_ref_np)
    pred = knn.predict(X_query[mask_q])
    acc = accuracy_score(y_query_eval, pred)
    return float(acc)


def transfer_accuracy_bidirectional(
    X_rna: np.ndarray,
    y_rna: Sequence[str],
    X_atac: np.ndarray,
    y_atac: Sequence[str],
    k: int = 5,
    metric: str = "cosine",
    l2_normalize: bool = True,
    restrict_to_intersection: bool = True,
    weights: str = "uniform",
) -> dict[str, float]:
    """
    Compute RNA->ATAC and ATAC->RNA transfer accuracy and their average.
    """
    acc_rna_as_ref = transfer_accuracy_embeddings(
        X_ref=X_rna,
        y_ref=y_rna,
        X_query=X_atac,
        y_query=y_atac,
        k=k,
        metric=metric,
        l2_normalize=l2_normalize,
        restrict_to_intersection=restrict_to_intersection,
        weights=weights,
    )
    acc_atac_as_ref = transfer_accuracy_embeddings(
        X_ref=X_atac,
        y_ref=y_atac,
        X_query=X_rna,
        y_query=y_rna,
        k=k,
        metric=metric,
        l2_normalize=l2_normalize,
        restrict_to_intersection=restrict_to_intersection,
        weights=weights,
    )
    avg = np.nanmean([acc_rna_as_ref, acc_atac_as_ref])
    return float(avg)


def transfer_accuracy(
    adata_combined,
    embed_key: str = "embedding",
    label_key: str = "cell type",
    domain_key: str = "domain",
    rna_name: str = "scRNA-seq",
    atac_name: str = "scATAC-seq",
    k: int = 5,
    metric: str = "cosine",
    l2_normalize: bool = True,
    restrict_to_intersection: bool = True,
    from_X: bool = False,
    weights: str = "uniform",
) -> dict[str, float]:
    """
    Convenience wrapper to compute bidirectional transfer accuracy directly
    from a single combined AnnData object.
    """

    if from_X:
        X = X = np.asarray(adata_combined.X)
    else:
        X = np.asarray(adata_combined.obsm[embed_key])

    y = adata_combined.obs[label_key].to_numpy()
    d = adata_combined.obs[domain_key].to_numpy()

    rna_idx = np.where(d == rna_name)[0]
    atac_idx = np.where(d == atac_name)[0]

    if len(rna_idx) == 0 or len(atac_idx) == 0:
        raise ValueError(
            f"No cells found for rna_name='{rna_name}' or atac_name='{atac_name}' "
            f"in obs['{domain_key}']."
        )

    X_rna, y_rna = X[rna_idx], y[rna_idx]
    X_atac, y_atac = X[atac_idx], y[atac_idx]

    return transfer_accuracy_bidirectional(
        X_rna,
        y_rna,
        X_atac,
        y_atac,
        k=k,
        metric=metric,
        l2_normalize=l2_normalize,
        restrict_to_intersection=restrict_to_intersection,
        weights=weights,
    )


"""--------------------Transfer accuracy end-----------------------"""


def _lisi_per_cell(
    x: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 30,
) -> np.ndarray:
    """
    Compute per-cell LISI = 1 / sum_c p_c^2 over the kNN label distribution.
    """
    x = np.asarray(x)
    labels = np.asarray(labels)
    n_cells = x.shape[0]

    if n_cells <= 1:
        return np.ones(n_cells, dtype=float)

    n_neighbors = int(n_neighbors)

    # +1 includes self, then drop it
    k = min(n_neighbors + 1, n_cells)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x)
    _, indices = nn.kneighbors(x)  # (n_cells, k)

    lisi = np.zeros(n_cells, dtype=float)
    for i in range(n_cells):
        neigh_idx = indices[i, 1:]  # drop self
        neigh_labels = labels[neigh_idx]

        if neigh_labels.size == 0:
            lisi[i] = 1.0
            continue

        _, counts = np.unique(neigh_labels, return_counts=True)
        p = counts.astype(float) / counts.sum()
        lisi[i] = 1.0 / np.sum(p**2)

    return lisi


def norm_lisi_label(
    x: np.ndarray,
    ct: np.ndarray,
    n_neighbors: int = 30,
) -> float:
    """
    cLISI-like score normalized to [0,1], higher=better (purer).
    Using MEAN aggregation (no median), less likely to saturate at 1.

    Steps:
      1) per-cell LISI on labels=ct
      2) per-cell-type MEAN LISI
      3) across cell types MEAN (macro-average)
      4) scale: LISI in [1, C] -> score in [0,1] (higher better)
    """
    x = np.asarray(x)
    ct = np.asarray(ct)
    n_neighbors = int(n_neighbors)

    labels = np.unique(ct)
    C = labels.size
    if C <= 1:
        return 1.0

    lisi = _lisi_per_cell(x, ct, n_neighbors=n_neighbors)

    # Macro aggregation: each cell type contributes equally
    per_type_mean = []
    for lab in labels:
        vals = lisi[ct == lab]
        if vals.size == 0:
            continue
        per_type_mean.append(float(vals.mean()))

    if len(per_type_mean) == 0:
        return float("nan")

    lisi_agg = float(np.mean(per_type_mean))  # mean across cell types

    # Scale to [0,1], higher better (purer)
    norm = (C - lisi_agg) / (C - 1.0)
    return float(np.clip(norm, 0.0, 1.0))


def norm_lisi_omics(
    x: np.ndarray,
    omics: np.ndarray,
    n_neighbors: int = 30,
) -> float:
    """
    iLISI-like score normalized to [0,1], higher=better (more mixed).
    Using MEAN aggregation (no median), less likely to saturate.

    Steps:
      1) per-cell LISI on labels=omics
      2) per-omics MEAN LISI
      3) across omics MEAN
      4) scale: LISI in [1, O] -> score in [0,1] (higher better)
    """
    x = np.asarray(x)
    omics = np.asarray(omics)
    n_neighbors = int(n_neighbors)

    omics_groups = np.unique(omics)
    O = omics_groups.size
    if O <= 1:
        return 1.0

    lisi = _lisi_per_cell(x, omics, n_neighbors=n_neighbors)

    per_omics_mean = []
    for o in omics_groups:
        vals = lisi[omics == o]
        if vals.size == 0:
            continue
        per_omics_mean.append(float(vals.mean()))

    if len(per_omics_mean) == 0:
        return float("nan")

    lisi_agg = float(np.mean(per_omics_mean))

    # Scale to [0,1], higher better (more mixed)
    norm = (lisi_agg - 1.0) / (O - 1.0)
    return float(np.clip(norm, 0.0, 1.0))
