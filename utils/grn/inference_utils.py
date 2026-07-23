from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

from .functions import edge_subgraph, window_graph


def motif_score_attr(left, right, distance):
    """Return the motif score stored in the BED score field."""
    try:
        score = float(right.score)
    except (ValueError, TypeError):
        score = 1.0
    return {"motif_score": score}


def load_or_compute_peak2tf(
    peak_bed,
    motif_bed,
    cache_path: str | os.PathLike,
    recompute: bool = False,
    require_motif_score: bool = True,
):
    """Load cached peak-to-TF motif overlaps or compute them from BED intervals.

    The cache is reused when it exists and contains the required edge
    attributes. If the cache is missing, or if ``require_motif_score=True`` and
    the cached graph has no ``motif_score`` edge attribute, the graph is
    recomputed and the cache is overwritten.
    """
    cache_path = Path(cache_path)
    should_compute = recompute or not cache_path.exists()

    if not should_compute:
        with cache_path.open("rb") as handle:
            peak2tf = pickle.load(handle)
        has_motif_score = (
            hasattr(peak2tf, "edge_attr_dict")
            and peak2tf.edge_attr_dict is not None
            and "motif_score" in peak2tf.edge_attr_dict
        )
        if require_motif_score and not has_motif_score:
            print(
                "Cached peak-to-TF graph has no motif_score; recomputing the cache."
            )
            should_compute = True
        else:
            print(f"Loaded cached peak-to-TF graph: {cache_path}")

    if should_compute:
        print("Computing peak-to-TF motif overlap. This may take a while.")
        peak2tf = window_graph(
            peak_bed,
            motif_bed,
            0,
            attr_fn=motif_score_attr,
            right_sorted=True,
        )
        with cache_path.open("wb") as handle:
            pickle.dump(peak2tf, handle)
        print(f"Saved peak-to-TF cache: {cache_path}")

    has_motif_score = (
        hasattr(peak2tf, "edge_attr_dict")
        and peak2tf.edge_attr_dict is not None
        and "motif_score" in peak2tf.edge_attr_dict
    )
    return peak2tf, has_motif_score


def filter_peak2tf_to_candidate_tfs(peak2tf, tfs):
    """Keep only peak-to-TF edges whose TF endpoint is in the candidate TF list."""
    tf_set = set(tfs)
    dst_indices = peak2tf.edge_index[1]
    mask = torch.tensor(
        [peak2tf.node_names[i] in tf_set for i in dst_indices.tolist()],
        dtype=torch.bool,
    )
    return edge_subgraph(peak2tf, mask)


def get_pruning_config(mode: str):
    """Return GRN pruning thresholds used by the tutorial."""
    prune_configs = {
        "strict": dict(importance_q=0.75, cis_top=500, require_peak=True),
        "relaxed": dict(importance_q=0.50, cis_top=500, require_peak=True),
    }
    if mode not in prune_configs:
        raise ValueError(
            f"Unknown PRUNE_MODE={mode!r}. Choose one of {list(prune_configs)}."
        )
    return prune_configs, prune_configs[mode]


def apply_importance_filter(network: pd.DataFrame, mode: str):
    """Filter GRNBoost2 edges by the importance quantile for a pruning mode."""
    prune_configs, config = get_pruning_config(mode)
    importance_threshold = network["importance"].quantile(config["importance_q"])
    network_filtered = network[network["importance"] >= importance_threshold].copy()
    return network_filtered, importance_threshold, prune_configs, config


def print_pruning_config(
    mode: str,
    config: dict,
    prune_configs: dict,
    importance_threshold: float,
    n_before: int,
    n_after: int,
) -> None:
    """Print the selected pruning thresholds in a compact tutorial format."""
    print(f"PRUNE_MODE = {mode!r}")
    print(
        "  "
        f"importance_q={config['importance_q']}, "
        f"cis_top={config['cis_top']}, "
        f"require_peak={config['require_peak']}"
    )
    print(
        f"Importance threshold ({config['importance_q']:.0%}): "
        f"{importance_threshold:.4f}"
    )
    print(f"After importance filtering: {n_before:,} -> {n_after:,} edges")

    print("\nAvailable pruning modes:")
    print(f"  {'mode':<10}  importance_q  cis_top  require_peak")
    for prune_mode, prune_config in prune_configs.items():
        marker = " <=" if prune_mode == mode else ""
        print(
            f"  {prune_mode:<10}  "
            f"{prune_config['importance_q']:.3f}         "
            f"{prune_config['cis_top']:<6}   "
            f"{prune_config['require_peak']}{marker}"
        )


def score_and_prune_grn(
    network_filtered: pd.DataFrame,
    gene2tf_rank: pd.DataFrame,
    g_sig,
    peak2tf_filtered,
    genes,
    peaks,
    tfs,
    cis_top_threshold: int,
    require_support_peak: bool,
    full_grn_file: str | os.PathLike,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Combine co-expression, cis ranking, and peak motif support into a GRN."""
    sig_pairs = set()
    for tf in gene2tf_rank.columns:
        top_genes = gene2tf_rank.index[gene2tf_rank[tf] <= cis_top_threshold]
        for gene in top_genes:
            sig_pairs.add((tf, gene))

    node_names = np.asarray(g_sig.node_names, dtype=object)
    edge_index = (
        g_sig.edge_index.detach().cpu().numpy()
        if torch.is_tensor(g_sig.edge_index)
        else np.asarray(g_sig.edge_index)
    )
    src, dst = edge_index[0], edge_index[1]

    score_arr = np.asarray(g_sig.edge_attr_dict["score"], dtype=float)
    qval_arr = np.asarray(g_sig.edge_attr_dict["qval"], dtype=float)

    gene_set = set(map(str, genes))
    peak_set = set(map(str, peaks))
    gene_peak_rows = []
    for i, (source, target) in enumerate(zip(src, dst)):
        source_name = str(node_names[source])
        target_name = str(node_names[target])
        if source_name in gene_set and target_name in peak_set:
            gene, peak = source_name, target_name
        elif target_name in gene_set and source_name in peak_set:
            gene, peak = target_name, source_name
        else:
            continue
        gene_peak_rows.append((gene, peak, score_arr[i], qval_arr[i]))

    gene_peak_df = pd.DataFrame(
        gene_peak_rows,
        columns=["target", "peak", "emb_score", "emb_qval"],
    )

    peak2tf_edge_index = (
        peak2tf_filtered.edge_index.detach().cpu().numpy()
        if torch.is_tensor(peak2tf_filtered.edge_index)
        else np.asarray(peak2tf_filtered.edge_index)
    )
    peak2tf_src, peak2tf_dst = peak2tf_edge_index[0], peak2tf_edge_index[1]
    peak2tf_names = np.asarray(peak2tf_filtered.node_names, dtype=object)
    tf_set = set(map(str, tfs))
    peak_tf_rows = []
    for source, target in zip(peak2tf_src, peak2tf_dst):
        source_name = str(peak2tf_names[source])
        target_name = str(peak2tf_names[target])
        if source_name in peak_set and target_name in tf_set:
            peak, tf = source_name, target_name
        elif target_name in peak_set and source_name in tf_set:
            peak, tf = target_name, source_name
        else:
            continue
        peak_tf_rows.append((peak, tf))

    peak_tf_df = pd.DataFrame(peak_tf_rows, columns=["peak", "TF"]).drop_duplicates()

    tf_target_support = gene_peak_df.merge(peak_tf_df, on="peak", how="inner")
    tf_target_support = tf_target_support.groupby(
        ["TF", "target"], as_index=False
    ).agg(
        n_support_peaks=("peak", "nunique"),
        emb_support=("emb_score", "max"),
        emb_qval_support=("emb_qval", "min"),
    )

    n_genes = gene2tf_rank.shape[0]
    cis_long = (
        gene2tf_rank.stack()
        .rename("cis_rank")
        .rename_axis(["target", "TF"])
        .reset_index()
    )
    cis_long["cis_percentile"] = (n_genes - cis_long["cis_rank"] + 1.0) / n_genes

    network_scored = (
        network_filtered.merge(cis_long, on=["TF", "target"], how="left")
        .merge(tf_target_support, on=["TF", "target"], how="left")
    )
    network_scored.to_csv(full_grn_file, index=False)

    network_pruned = network_scored[
        network_scored.apply(lambda row: (row["TF"], row["target"]) in sig_pairs, axis=1)
    ].copy()
    if require_support_peak:
        network_pruned = network_pruned[
            network_pruned["n_support_peaks"].fillna(0) > 0
        ].copy()

    network_pruned["emb_sig_support"] = -np.log10(
        np.clip(
            network_pruned["emb_qval_support"].to_numpy(dtype=float),
            1e-300,
            1.0,
        )
    )

    def rank01(series: pd.Series) -> pd.Series:
        return series.rank(method="average", pct=True)

    network_pruned["r_importance"] = rank01(network_pruned["importance"])
    network_pruned["r_emb"] = rank01(network_pruned["emb_sig_support"])
    network_pruned["r_cis"] = rank01(network_pruned["cis_percentile"])

    w_importance, w_emb, w_cis = 0.40, 0.25, 0.35
    network_pruned["final_reg_score"] = (
        w_importance * network_pruned["r_importance"]
        + w_emb * network_pruned["r_emb"]
        + w_cis * network_pruned["r_cis"]
    )
    network_pruned = network_pruned.sort_values(
        "final_reg_score",
        ascending=False,
    ).reset_index(drop=True)

    return network_scored, network_pruned, len(sig_pairs)


def save_and_summarize_pruned_grn(
    network: pd.DataFrame,
    network_pruned: pd.DataFrame,
    output_file: str | os.PathLike,
) -> None:
    """Save the pruned GRN and print a compact summary."""
    network_pruned.to_csv(output_file, index=False)

    print("\n" + "=" * 50)
    print("         Final Pruned GRN Summary")
    print("=" * 50)
    print(f"  Original edges     : {len(network):>8,}")
    print(f"  Pruned edges       : {len(network_pruned):>8,}")
    print(f"  Unique TFs         : {network_pruned['TF'].nunique():>8,}")
    print(f"  Unique targets     : {network_pruned['target'].nunique():>8,}")
    print(
        f"  Importance range   : {network_pruned['importance'].min():.4f} ~ "
        f"{network_pruned['importance'].max():.4f}"
    )
    print(
        f"  Final score range  : {network_pruned['final_reg_score'].min():.4f} ~ "
        f"{network_pruned['final_reg_score'].max():.4f}"
    )
    print(f"  Saved to           : {output_file}")
    print("=" * 50)


def plot_selected_tf_network(
    grn: pd.DataFrame,
    rna,
    selected_tfs: list[str],
    plot_file: str | os.PathLike,
    plot_pdf_file: str | os.PathLike,
    top_genes_per_tf: int = 20,
    min_final_score: float | None = None,
    edge_width: float = 1.8,
    label_fontsize: int = 12,
    export_dpi: int = 1200,
) -> tuple[list[str], list[str]]:
    """Plot selected TF-to-target links as a radial network."""
    import scanpy as sc

    if min_final_score is not None:
        grn = grn[grn["final_reg_score"] >= min_final_score].copy()

    available_tfs = set(grn["TF"].unique())
    tfs_to_plot = [tf for tf in selected_tfs if tf in available_tfs]
    missing_tfs = [tf for tf in selected_tfs if tf not in available_tfs]

    if not tfs_to_plot:
        raise ValueError("None of the selected TFs were found in pruned_grn.csv")

    if missing_tfs:
        print(f"Skipped TFs not found: {missing_tfs}")

    fig = plt.figure(figsize=(9.5, 8.2))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.05, right=0.84, top=0.90, bottom=0.06)

    edge_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "OrRd_trim",
        plt.cm.OrRd(np.linspace(0.30, 1.00, 256)),
    )
    gene_cmap = plt.cm.viridis
    all_scores = []
    edge_rows = []

    for tf in tfs_to_plot:
        df_tf = (
            grn[grn["TF"] == tf]
            .sort_values("final_reg_score", ascending=False)
            .head(top_genes_per_tf)
            .copy()
        )
        df_tf["TF"] = tf
        edge_rows.append(df_tf)
        all_scores.extend(df_tf["final_reg_score"].tolist())

    plot_df = pd.concat(edge_rows, ignore_index=True)
    gene_score = (
        plot_df.groupby("target")["final_reg_score"].max().sort_values(ascending=False)
    )
    genes = gene_score.index.tolist()

    rna_expr = rna.copy()
    if "counts" in rna_expr.layers:
        rna_expr.X = rna_expr.layers["counts"].copy()
    sc.pp.normalize_total(rna_expr, target_sum=1e4)
    sc.pp.log1p(rna_expr)
    gene_mean_expr = pd.Series(
        np.asarray(rna_expr.X.mean(axis=0)).ravel(),
        index=rna_expr.var_names,
    )
    gene_expr_vals = (
        gene_mean_expr.reindex(genes)
        .fillna(gene_mean_expr.median())
        .to_numpy(dtype=float)
    )
    tf_expr_vals = (
        gene_mean_expr.reindex(tfs_to_plot)
        .fillna(gene_mean_expr.median())
        .to_numpy(dtype=float)
    )

    score_min = float(np.min(all_scores))
    score_max = float(np.max(all_scores))
    if score_max == score_min:
        score_max = score_min + 1e-8
    score_norm = mpl.colors.Normalize(vmin=score_min, vmax=score_max)

    expr_lo = float(np.percentile(gene_expr_vals, 5))
    expr_hi = float(np.percentile(gene_expr_vals, 95))
    if expr_hi <= expr_lo:
        expr_hi = expr_lo + 1e-8
    expr_norm = mpl.colors.Normalize(vmin=expr_lo, vmax=expr_hi)

    gene_theta = np.linspace(
        np.pi / 2,
        np.pi / 2 - 2 * np.pi,
        len(genes),
        endpoint=False,
    )
    gene_r = 1.1
    gene_pos = {
        gene: (gene_r * np.cos(angle), gene_r * np.sin(angle), angle)
        for gene, angle in zip(genes, gene_theta)
    }

    if len(tfs_to_plot) == 1:
        tf_angles = np.array([np.pi / 2])
    else:
        tf_angles = np.linspace(
            np.pi / 2 - np.pi / 4,
            np.pi / 2 + np.pi / 4,
            len(tfs_to_plot),
        )
    tf_r = 0.55
    tf_pos = {
        tf: (tf_r * np.cos(angle), tf_r * np.sin(angle))
        for tf, angle in zip(tfs_to_plot, tf_angles)
    }

    for _, row in plot_df.iterrows():
        tf = row["TF"]
        target = row["target"]
        score = float(row["final_reg_score"])
        x0, y0 = tf_pos[tf]
        x1, y1, _ = gene_pos[target]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=edge_cmap(score_norm(score)),
            lw=edge_width,
            linestyle="-",
            alpha=1.0,
            solid_capstyle="round",
            zorder=1,
        )

    gene_colors = gene_cmap(expr_norm(gene_expr_vals))
    ax.scatter(
        [gene_pos[gene][0] for gene in genes],
        [gene_pos[gene][1] for gene in genes],
        s=290,
        c=gene_colors,
        marker="o",
        edgecolor="#333333",
        linewidth=0.9,
        zorder=3,
    )

    tf_colors = gene_cmap(expr_norm(tf_expr_vals))
    ax.scatter(
        [tf_pos[tf][0] for tf in tfs_to_plot],
        [tf_pos[tf][1] for tf in tfs_to_plot],
        s=320,
        c=tf_colors,
        marker="s",
        edgecolor="#222222",
        linewidth=1.0,
        zorder=4,
    )
    for tf in tfs_to_plot:
        x, y = tf_pos[tf]
        ax.text(
            x,
            y + 0.08,
            tf,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for gene in genes:
        _, _, angle = gene_pos[gene]
        text_x, text_y = 1.18 * np.cos(angle), 1.18 * np.sin(angle)
        degrees = np.degrees(angle)
        if -90 <= degrees <= 90:
            rotation = degrees
            horizontal_alignment = "left"
        else:
            rotation = degrees + 180
            horizontal_alignment = "right"
        ax.text(
            text_x,
            text_y,
            gene,
            fontsize=label_fontsize,
            rotation=rotation,
            rotation_mode="anchor",
            ha=horizontal_alignment,
            va="center",
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.axis("off")

    cax_edge = fig.add_axes([0.865, 0.56, 0.020, 0.24])
    cax_gene = fig.add_axes([0.865, 0.24, 0.020, 0.24])

    sm_edge = mpl.cm.ScalarMappable(norm=score_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar_edge = fig.colorbar(sm_edge, cax=cax_edge)
    cbar_edge.set_label("Edge: final_reg_score", fontsize=11)
    cbar_edge.ax.tick_params(labelsize=10)

    sm_gene = mpl.cm.ScalarMappable(norm=expr_norm, cmap=gene_cmap)
    sm_gene.set_array([])
    cbar_gene = fig.colorbar(sm_gene, cax=cax_gene)
    cbar_gene.set_label("Node: mean log1p expression", fontsize=11)
    cbar_gene.ax.tick_params(labelsize=10)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#222222",
            markeredgewidth=1.2,
            markersize=9,
            label="TF",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#333333",
            markeredgewidth=1.2,
            markersize=9,
            label="Target",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )

    fig.suptitle("Selected TF-Gene Subnetwork", y=0.975, fontsize=14)
    fig.savefig(plot_pdf_file, dpi=export_dpi)
    plt.show()

    print(f"Saved selected TF network plot to: {plot_pdf_file}")
    return tfs_to_plot, missing_tfs
