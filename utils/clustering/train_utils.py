"""Training utilities for single-run multi-omics clustering notebooks."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.loader import HGTLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_view_edge_index_dict(batch, keep_modality: str):
    """
    Return a filtered edge_index_dict for the given modality view.
    Does NOT copy or modify the original batch.

    keep_modality:
        - 'rna':  only cell-gene / gene-cell edges
        - 'atac': only cell-peak / peak-cell edges
        - 'full': all edges (returns batch.edge_index_dict directly)
    """
    if keep_modality == "full":
        return batch.edge_index_dict

    result = {}
    for edge_type, edge_index in batch.edge_index_dict.items():
        src, _, dst = edge_type
        if (
            keep_modality == "rna"
            and "gene" in (src, dst)
            or keep_modality == "atac"
            and "peak" in (src, dst)
        ):
            result[edge_type] = edge_index
        else:
            result[edge_type] = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    return result


def split_x_by_modality(x_dict, modality: str, rna_dim: int):
    """
    Mask cell feature dimensions according to the selected graph view.

    Cell feature layout:
        [RNA PCA dims | ATAC LSI dims]

    RNA view:
        keep RNA dims, zero ATAC dims.
    ATAC view:
        zero RNA dims, keep ATAC dims.
    Full view:
        keep both.
    """
    out = {}
    for nt, x in x_dict.items():
        if nt != "cell":
            out[nt] = x
            continue
        x_new = x.clone()
        if modality == "rna":
            x_new[:, rna_dim:] = 0
        elif modality == "atac":
            x_new[:, :rna_dim] = 0
        elif modality == "full":
            pass
        else:
            raise ValueError(f"Unknown modality: {modality}")
        out[nt] = x_new
    return out


def feature_mask(x_dict, mask_ratio: float = 0.1):
    """Random feature masking as lightweight augmentation."""
    if mask_ratio <= 0:
        return x_dict
    out = {}
    for nt, x in x_dict.items():
        mask = torch.rand_like(x) > mask_ratio
        out[nt] = x * mask.float()
    return out


def get_target_size(batch):
    """HGTLoader places target input_nodes at the beginning of batch['cell']."""
    return int(getattr(batch["cell"], "batch_size", batch["cell"].x.size(0)))


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    rna_dim: int,
    feat_mask_ratio: float = 0.1,
    grad_clip: float = 1.0,
):
    model.train()
    total = {
        "loss_total": 0.0,
        "loss_ra": 0.0,
        "loss_fr": 0.0,
        "loss_fa": 0.0,
        "pos_ra": 0.0,
        "neg_ra": 0.0,
        "pos_fr": 0.0,
        "neg_fr": 0.0,
        "pos_fa": 0.0,
        "neg_fa": 0.0,
    }
    n_batch = 0

    for batch in loader:
        batch = batch.to(device)
        target_size = get_target_size(batch)

        edge_index_dict_rna = get_view_edge_index_dict(batch, "rna")
        edge_index_dict_atac = get_view_edge_index_dict(batch, "atac")
        edge_index_dict_full = batch.edge_index_dict

        x_rna = split_x_by_modality(batch.x_dict, modality="rna", rna_dim=rna_dim)
        x_atac = split_x_by_modality(batch.x_dict, modality="atac", rna_dim=rna_dim)
        x_full = split_x_by_modality(batch.x_dict, modality="full", rna_dim=rna_dim)

        x_rna = feature_mask(x_rna, mask_ratio=feat_mask_ratio)
        x_atac = feature_mask(x_atac, mask_ratio=feat_mask_ratio)
        x_full = feature_mask(x_full, mask_ratio=feat_mask_ratio)

        loss, stats = model(
            x_rna,
            edge_index_dict_rna,
            x_atac,
            edge_index_dict_atac,
            x_full,
            edge_index_dict_full,
            target_size=target_size,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in total:
            total[k] += stats[k]
        n_batch += 1

    return {k: v / max(n_batch, 1) for k, v in total.items()}


@torch.no_grad()
def extract_all_cell_embeddings(model, data, device, batch_size: int = 512):
    """Extract final full-graph cell embeddings for all cells."""
    model.eval()
    n_cell = data["cell"].num_nodes
    hidden_embs = torch.zeros(n_cell, model.encoder.hidden_dim)
    proj_embs = torch.zeros(n_cell, model.encoder.out_dim)
    seen = torch.zeros(n_cell, dtype=torch.bool)

    loader = HGTLoader(
        data,
        num_samples={"cell": [512, 256], "gene": [512, 256], "peak": [1024, 512]},
        input_nodes="cell",
        batch_size=batch_size,
        shuffle=False,
    )

    for batch in loader:
        batch = batch.to(device)
        target_size = get_target_size(batch)
        h_dict, z_cell = model.encoder(model._gate_full_x_dict(batch.x_dict), batch.edge_index_dict)
        original_ids = batch["cell"].n_id[:target_size].cpu()
        hidden_embs[original_ids] = h_dict["cell"][:target_size].cpu()
        proj_embs[original_ids] = z_cell[:target_size].cpu()
        seen[original_ids] = True

    if not seen.all():
        missing = int((~seen).sum().item())
        print(f"[Warning] {missing} cells were not assigned embeddings; they remain zeros.")
    return hidden_embs.numpy(), proj_embs.numpy()


def evaluate_clustering(embeddings, n_clusters_range=(8, 15, 25), seed: int = 42):
    """Fast unsupervised sanity check using KMeans silhouette score."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = embeddings.shape[0]
    if n < 3:
        return {}
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(n, size=min(5000, n), replace=False)
    sample = embeddings[sample_idx]

    results = {}
    for k in n_clusters_range:
        if k >= sample.shape[0]:
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(sample)
        if len(np.unique(labels)) > 1:
            results[k] = float(silhouette_score(sample, labels))
    return results
