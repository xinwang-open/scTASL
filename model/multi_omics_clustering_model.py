"""
HGT encoder + batch-level symmetric InfoNCE for paired single-cell multi-omics clustering.

Main design:
    - Shared HGT encoder for all graph views.
    - RNA view: cell-gene subgraph with RNA cell features.
    - ATAC view: cell-peak subgraph with ATAC cell features.
    - Full view: cell-gene-peak graph with concatenated RNA+ATAC cell features.
    - Symmetric in-batch InfoNCE aligns paired cells across views.
    - InfoNCE can be applied directly on HGT hidden states or projected states.
    - Lightweight edge reconstruction keeps cell-gene/cell-peak graph structure predictive.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import HGTLoader
from torch_geometric.nn import HGTConv, Linear

from utils.clustering.data import prepare_data
from utils.clustering.train_utils import (
    evaluate_clustering,
    extract_all_cell_embeddings,
    set_seed,
    train_one_epoch,
)


class CrossModalGate(nn.Module):
    """
    Per-cell soft gate for full-view cell features.

    Learns a per-cell (rna_weight, atac_weight) via softmax so cells with
    sparse or noisy ATAC automatically down-weight that modality and rely
    more on RNA, and vice versa.  Gates are scaled by 2 so their mean is
    1.0, preserving average feature magnitude for the downstream linear layer.
    """

    def __init__(self, rna_dim: int, atac_dim: int):
        super().__init__()
        self.rna_dim = rna_dim
        self.atac_dim = atac_dim
        in_dim = rna_dim + atac_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rna = x[:, : self.rna_dim]
        x_atac = x[:, self.rna_dim :]
        gates = 2.0 * torch.softmax(self.net(x), dim=-1)  # [N, 2], mean gate = 1.0
        return torch.cat([gates[:, 0:1] * x_rna, gates[:, 1:2] * x_atac], dim=-1)


class SingleCellHGT(nn.Module):
    """Heterogeneous graph Transformer encoder for cell-gene-peak graphs."""

    def __init__(
        self,
        metadata,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        out_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lin_dict = nn.ModuleDict()
        self.norm_input = nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)
            self.norm_input[node_type] = nn.LayerNorm(hidden_dim)

        self.convs = nn.ModuleList()
        self.norms_after = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads))
            self.norms_after.append(
                nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in metadata[0]})
            )

        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x_dict, edge_index_dict):
        h_dict = {}
        for node_type, x in x_dict.items():
            h = self.lin_dict[node_type](x)
            h = self.norm_input[node_type](h)
            h_dict[node_type] = F.relu(h)

        for conv, norm_dict in zip(self.convs, self.norms_after):
            h_new = conv(h_dict, edge_index_dict)
            updated = {}
            for nt in h_dict:
                if nt in h_new and h_new[nt] is not None:
                    updated[nt] = norm_dict[nt](h_dict[nt] + self.dropout(F.leaky_relu(h_new[nt])))
                else:
                    updated[nt] = h_dict[nt]
            h_dict = updated
        return h_dict

    def forward(self, x_dict, edge_index_dict):
        h_dict = self.encode(x_dict, edge_index_dict)
        z_cell = self.projector(h_dict["cell"])
        z_cell = F.normalize(z_cell, dim=-1)
        return h_dict, z_cell


class TriViewInfoNCE(nn.Module):
    """
    Shared-HGT tri-view contrastive model.

    Loss:
        L = L(RNA, ATAC) + lambda_full * 0.5 * [L(Full, RNA) + L(Full, ATAC)]

    The full view passes cell features through CrossModalGate before encoding,
    so the model learns per-cell RNA/ATAC reliability weights.
    """

    def __init__(
        self,
        metadata,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.2,
        lambda_full: float = 0.5,
        rna_dim: int = 50,
        atac_dim: int | None = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_full = lambda_full
        _atac_dim = atac_dim if atac_dim is not None else rna_dim
        self.cross_modal_gate = CrossModalGate(rna_dim, _atac_dim)
        self.encoder = SingleCellHGT(
            metadata=metadata,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def _gate_full_x_dict(self, x_dict: dict) -> dict:
        """Apply cross-modal gate to cell features; leave gene/peak features unchanged."""
        out = dict(x_dict)
        out["cell"] = self.cross_modal_gate(x_dict["cell"])
        return out

    def info_nce(self, z1: torch.Tensor, z2: torch.Tensor):
        """Symmetric in-batch InfoNCE. Same index is the positive pair."""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = (z1 @ z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss_12 = F.cross_entropy(logits, labels)
        loss_21 = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_12 + loss_21)

    @staticmethod
    def _similarity_stats(z1: torch.Tensor, z2: torch.Tensor):
        with torch.no_grad():
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            sim = z1 @ z2.T
            pos = sim.diag().mean().item()
            if sim.size(0) > 1:
                neg_mask = ~torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
                neg = sim[neg_mask].mean().item()
            else:
                neg = 0.0
        return pos, neg

    def forward(
        self,
        x_dict_rna,
        edge_index_dict_rna,
        x_dict_atac,
        edge_index_dict_atac,
        x_dict_full,
        edge_index_dict_full,
        target_size: int | None = None,
    ):
        _, z_rna = self.encoder(x_dict_rna, edge_index_dict_rna)
        _, z_atac = self.encoder(x_dict_atac, edge_index_dict_atac)
        _, z_full = self.encoder(self._gate_full_x_dict(x_dict_full), edge_index_dict_full)

        if target_size is not None:
            z_rna = z_rna[:target_size]
            z_atac = z_atac[:target_size]
            z_full = z_full[:target_size]

        loss_ra = self.info_nce(z_rna, z_atac)
        loss_fr = self.info_nce(z_full, z_rna)
        loss_fa = self.info_nce(z_full, z_atac)
        loss = loss_ra + self.lambda_full * 0.5 * (loss_fr + loss_fa)

        pos_ra, neg_ra = self._similarity_stats(z_rna, z_atac)
        pos_fr, neg_fr = self._similarity_stats(z_full, z_rna)
        pos_fa, neg_fa = self._similarity_stats(z_full, z_atac)

        stats = {
            "loss_total": float(loss.detach().cpu()),
            "loss_ra": float(loss_ra.detach().cpu()),
            "loss_fr": float(loss_fr.detach().cpu()),
            "loss_fa": float(loss_fa.detach().cpu()),
            "pos_ra": pos_ra,
            "neg_ra": neg_ra,
            "pos_fr": pos_fr,
            "neg_fr": neg_fr,
            "pos_fa": pos_fa,
            "neg_fa": neg_fa,
        }
        return loss, stats

    @torch.no_grad()
    def extract_cell_embedding(self, x_dict, edge_index_dict):
        _, z_cell = self.encoder(self._gate_full_x_dict(x_dict), edge_index_dict)
        return z_cell


class MultiOmicsClusteringModel(TriViewInfoNCE):
    """Notebook-facing interface for a single multi-omics clustering run."""

    def __init__(
        self,
        data,
        result_path: str,
        hidden_dim: int = 128,
        out_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.2,
        lambda_full: float = 0.5,
        batch_size: int = 256,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        feature_mask_ratio: float = 0.1,
        gradient_clip: float = 1.0,
        evaluate_every: int = 10,
        print_every: int = 10,
        seed: int = 42,
        device: str | torch.device | None = None,
    ):
        set_seed(seed)
        rna_dim = int(data["cell"].rna_dim)
        atac_dim = int(data["cell"].atac_dim)
        super().__init__(
            metadata=data.metadata(),
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            temperature=temperature,
            lambda_full=lambda_full,
            rna_dim=rna_dim,
            atac_dim=atac_dim,
        )

        self.data = data
        self.result_path = result_path
        self.rna_dim = rna_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.feature_mask_ratio = feature_mask_ratio
        self.gradient_clip = gradient_clip
        self.evaluate_every = evaluate_every
        self.print_every = max(1, int(print_every))
        self.seed = seed
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.to(self.device)

        os.makedirs(self.result_path, exist_ok=True)
        self.config = {
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "temperature": temperature,
            "lambda_full": lambda_full,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "feature_mask_ratio": feature_mask_ratio,
            "gradient_clip": gradient_clip,
            "evaluate_every": evaluate_every,
            "print_every": self.print_every,
            "seed": seed,
        }
        with open(
            os.path.join(self.result_path, "model_config.json"), "w", encoding="utf-8"
        ) as file:
            json.dump(self.config, file, indent=2, ensure_ascii=False)

        self._initialize_lazy_layers()
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=learning_rate * 0.01,
        )
        self.train_loader = HGTLoader(
            self.data,
            num_samples={"cell": [512, 256], "gene": [512, 256], "peak": [1024, 512]},
            input_nodes="cell",
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.training_log = None
        self.final_hidden = None
        self.final_projected = None

    @staticmethod
    def data_processing(
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
        """Load paired modalities and construct the heterogeneous graph."""
        return prepare_data(
            rna_path,
            atac_path,
            n_hvg=n_hvg,
            n_comp=n_comp,
            k_gene=k_gene,
            k_peak=k_peak,
            min_cells=min_cells,
            min_peak_frac=min_peak_frac,
            max_peak_frac=max_peak_frac,
        )

    def _initialize_lazy_layers(self):
        dummy_batch = next(
            iter(
                HGTLoader(
                    self.data,
                    num_samples={"cell": [8, 4], "gene": [8, 4], "peak": [8, 4]},
                    input_nodes="cell",
                    batch_size=min(4, self.data["cell"].num_nodes),
                    shuffle=False,
                )
            )
        ).to(self.device)
        with torch.no_grad():
            self.encoder(dummy_batch.x_dict, dummy_batch.edge_index_dict)

    def train_model(self):
        """Train once using the configuration supplied at initialization."""
        best_silhouette = -1.0
        rows = []

        for epoch in range(1, self.epochs + 1):
            stats = train_one_epoch(
                model=self,
                loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                rna_dim=self.rna_dim,
                feat_mask_ratio=self.feature_mask_ratio,
                grad_clip=self.gradient_clip,
            )
            self.scheduler.step()

            embedding_std = np.nan
            best_k = -1
            silhouette = np.nan
            if epoch % self.evaluate_every == 0 or epoch == self.epochs:
                _, projected = extract_all_cell_embeddings(
                    self,
                    self.data,
                    self.device,
                    batch_size=self.batch_size,
                )
                embedding_std = float(projected.std(axis=0).mean())
                silhouette_results = evaluate_clustering(projected, seed=self.seed)
                if silhouette_results:
                    best_k = max(silhouette_results, key=silhouette_results.get)
                    silhouette = silhouette_results[best_k]
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        np.save(
                            os.path.join(self.result_path, "best_cell_embedding.npy"), projected
                        )
                        torch.save(
                            {
                                "model_state": self.state_dict(),
                                "epoch": epoch,
                                "config": self.config,
                            },
                            os.path.join(self.result_path, "best_model.pt"),
                        )

            row = {"epoch": epoch, **stats, "learning_rate": self.optimizer.param_groups[0]["lr"]}
            row.update({"embedding_std": embedding_std, "best_k": best_k, "silhouette": silhouette})
            rows.append(row)
            if epoch == 1 or epoch % self.print_every == 0 or epoch == self.epochs:
                print(
                    f"Epoch {epoch:03d}/{self.epochs:03d} | "
                    f"loss={stats['loss_total']:.4f} "
                    f"ra={stats['loss_ra']:.4f} fr={stats['loss_fr']:.4f} "
                    f"fa={stats['loss_fa']:.4f} lr={row['learning_rate']:.2e}"
                )

        self.training_log = pd.DataFrame(rows)
        self.training_log.to_csv(
            os.path.join(self.result_path, "train_log.tsv"),
            sep="\t",
            index=False,
        )
        return self.training_log

    def encode_data(self, rna, embedding_key: str = "X_hgt_infonce"):
        """Attach learned embeddings to RNA AnnData and save final artifacts."""
        self.final_hidden, self.final_projected = extract_all_cell_embeddings(
            self,
            self.data,
            self.device,
            batch_size=self.batch_size,
        )
        np.save(
            os.path.join(self.result_path, "final_cell_embedding_hidden.npy"),
            self.final_hidden,
        )
        np.save(
            os.path.join(self.result_path, "final_cell_embedding_proj.npy"),
            self.final_projected,
        )

        rna.obsm[embedding_key] = self.final_projected
        rna.obsm["X_hgt_proj"] = self.final_projected
        rna.obsm["X_hgt_hidden"] = self.final_hidden
        rna.write_h5ad(os.path.join(self.result_path, "final_rna_with_embedding.h5ad"))
        torch.save(
            {"model_state": self.state_dict(), "epoch": self.epochs, "config": self.config},
            os.path.join(self.result_path, "final_model.pt"),
        )
        return rna
