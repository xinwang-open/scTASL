import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

palette30 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#3b3fb4",
    "#637939",
    "#C94E28",
    "#0B4F6C",
    "#9e1f8d",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#878acd",
    "#89935a",
    "#DF7555",
    "#BEE9F6",
    "#de9ed6",
]


def cell_type_palette(adata, label_key: str = "cell_type"):
    """Return a Scanpy palette for a categorical cell-type annotation."""
    adata.obs[label_key] = adata.obs[label_key].astype("category")
    n_cell_types = adata.obs[label_key].nunique()
    if n_cell_types <= 20:
        return None
    if n_cell_types <= len(palette30):
        return palette30[:n_cell_types]
    return sns.color_palette("husl", n_cell_types).as_hex()


def knn_transfer_confusion_rna_to_atac_percent(
    adata_combined,
    embed_key: str = "embedding",
    label_key: str = "cell_type",
    domain_key: str = "domain",
    rna_name: str = "scRNA-seq",
    atac_name: str = "scATAC-seq",
    k: int = 11,
    metric: str = "cosine",
    weights: str = "uniform",
    l2_normalize: bool = True,
    restrict_to_intersection: bool = True,
    percent: bool = True,  # 显示为百分比
    plot: bool = True,
    heatmap_cmap: str = "Oranges",  # 类似示例的黄绿渐变
    save_path: str = None,
):
    # 取嵌入与元信息
    X = np.asarray(adata_combined.obsm[embed_key])
    y = adata_combined.obs[label_key].to_numpy()
    d = adata_combined.obs[domain_key].to_numpy()

    rna_idx = np.where(d == rna_name)[0]
    atac_idx = np.where(d == atac_name)[0]
    if len(rna_idx) == 0 or len(atac_idx) == 0:
        raise ValueError("检查 domain 名称是否与 rna_name/atac_name 一致。")

    X_rna, y_rna = X[rna_idx], y[rna_idx]
    X_atac, y_atac = X[atac_idx], y[atac_idx]

    if l2_normalize:
        X_rna = normalize(X_rna, norm="l2", axis=1)
        X_atac = normalize(X_atac, norm="l2", axis=1)

    if restrict_to_intersection:
        ref_set = set(y_rna)
        mask_atac = np.array([lab in ref_set for lab in y_atac], dtype=bool)
        X_atac_eval = X_atac[mask_atac]
        y_true = y_atac[mask_atac]
    else:
        X_atac_eval = X_atac
        y_true = y_atac

    if X_atac_eval.shape[0] == 0:
        raise ValueError("筛选后 ATAC 无可评样本，请检查标签或设 restrict_to_intersection=False。")

    # 训练 kNN（RNA 为参考）并预测
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
    knn.fit(X_rna, y_rna)
    y_pred = knn.predict(X_atac_eval)

    labels_order = np.unique(y_true)
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_order, normalize=None)

    # 行归一化 -> 百分比
    if percent:
        row_sums = cm_counts.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0
        cm_pct = (cm_counts / row_sums) * 100.0
        cm_df = pd.DataFrame(np.round(cm_pct, 2), index=labels_order, columns=labels_order)
    else:
        cm_df = pd.DataFrame(cm_counts, index=labels_order, columns=labels_order)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels_order, zero_division=0)

    if plot:
        mat = cm_df.values
        fig, ax = plt.subplots(
            figsize=(max(6, 0.6 * len(labels_order)), max(5, 0.6 * len(labels_order)))
        )
        im = ax.imshow(
            mat, aspect="auto", interpolation="nearest", cmap=heatmap_cmap, vmin=0, vmax=100
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("value", rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(labels_order)))
        ax.set_xticklabels(labels_order, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels_order)))
        ax.set_yticklabels(labels_order)
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(np.arange(mat.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(mat.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.grid(which="major", color="white", linestyle="", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_title("RNA→ATAC Confusion Matrix")
        ax.set_xlabel("Transferred labels")
        ax.set_ylabel("Original labels")

        text_threshold = mat.max() * 0.5 if mat.size else 0
        for i in range(len(labels_order)):
            for j in range(len(labels_order)):
                val = mat[i, j]
                txt = f"{val:.1f}" if percent else f"{int(val)}"
                color = "white" if val >= text_threshold else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=1200, bbox_inches="tight")
        plt.show()

    return y_true, y_pred, cm_df, report, acc
