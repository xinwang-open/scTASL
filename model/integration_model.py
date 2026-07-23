import collections
import itertools
import os
from math import ceil

import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as sched

from utils.common.config import config, normalize_edges
from utils.common.dataloader import (
    AnnDataset,
    DataLoader,
    GraphDataset,
    ParallelDataLoader,
)
from utils.common.losses import info_nce_cross

from .modules import (
    Decoder_g,
    Discriminator,
    Encoder,
    GraphDecoder,
    GraphEncoder,
    Predictor,
    Prior,
)


class Integration_Model(nn.Module):
    def __init__(
        self,
        data_config: dict,
        graph: GraphDataset,
        vertices: list,
        is_paired: bool = False,
        emb_size: int = 256,
        latent_dim: int = 64,
        dropout_rate: float = 0.2,
        lr: float = 2e-3,
        shared_batches: bool = False,
        lam_kl: float = 0.5,
        lam_graph: float = 0.02,
        lam_d: float = 0.02,
        result_path: collections.OrderedDict | None = None,
    ):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.data_config = data_config
        self.is_paired = bool(is_paired)
        self.vertices = pd.Index(vertices)
        self.number_vertices = self.vertices.size
        self.lr = lr
        self.lam_kl = lam_kl
        self.lam_graph = lam_graph
        self.lam_d = lam_d
        self.result_path = result_path
        self.log_path = None
        self.early_stop_patience = 200
        self.no_improve_count = 0
        self.best_val_vae_loss = float("inf")

        self.eidx = torch.as_tensor(graph.eidx, dtype=torch.long, device=self.device)
        self.ewt = torch.as_tensor(graph.ewt, dtype=torch.float, device=self.device)
        self.enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt), dtype=torch.float32, device=self.device
        )

        self.data_config["RNA"]["batches"] = (
            pd.Index([])
            if data_config["RNA"]["batches"] is None
            else pd.Index(data_config["RNA"]["batches"])
        )
        self.data_config["ATAC"]["batches"] = (
            pd.Index([])
            if data_config["ATAC"]["batches"] is None
            else pd.Index(data_config["ATAC"]["batches"])
        )
        self.data_config["RNA"]["features"] = pd.Index(self.data_config["RNA"]["features"])
        self.data_config["ATAC"]["features"] = pd.Index(self.data_config["ATAC"]["features"])

        if shared_batches:
            rna_batches = data_config["RNA"]["batches"]
            atac_batches = data_config["ATAC"]["batches"]
            if not np.array_equal(rna_batches, atac_batches):
                raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = rna_batches.size
        else:
            du_n_batches = 0

        self.rna_encoder = Encoder(
            input_dim=len(data_config["RNA"]["features"]),
            emb_size=emb_size,
            output_dim=latent_dim,
            dropout_rate=dropout_rate,
        ).to(self.device)
        self.atac_encoder = Encoder(
            input_dim=len(data_config["ATAC"]["features"]),
            emb_size=emb_size,
            output_dim=latent_dim,
            dropout_rate=dropout_rate,
        ).to(self.device)

        self.rna_decoder = Decoder_g(
            out_dim=len(data_config["RNA"]["features"]),
            n_batches=max(1, data_config["RNA"]["batches"].size),
        ).to(self.device)
        self.atac_decoder = Decoder_g(
            out_dim=len(data_config["ATAC"]["features"]),
            n_batches=max(1, data_config["ATAC"]["batches"].size),
        ).to(self.device)

        self.graph_encoder = GraphEncoder(
            vnum=self.number_vertices,
            out_dim=latent_dim,
        ).to(self.device)
        self.graph_decoder = GraphDecoder().to(self.device)
        self.d_discriminator = Discriminator(
            input_dim=latent_dim,
            output_dim=2,
            n_batches=du_n_batches,
            dropout_rate=dropout_rate,
        ).to(self.device)
        self.predictor = Predictor(input_dim=latent_dim, output_dim=32).to(self.device)
        self.prior = Prior().to(self.device)

        self.optim_G = torch.optim.RMSprop(
            itertools.chain(
                self.rna_encoder.parameters(),
                self.atac_encoder.parameters(),
                self.rna_decoder.parameters(),
                self.atac_decoder.parameters(),
                self.graph_encoder.parameters(),
                self.graph_decoder.parameters(),
                self.predictor.parameters(),
            ),
            lr=self.lr,
            alpha=0.95,
        )
        self.optim_D = torch.optim.RMSprop(
            self.d_discriminator.parameters(), lr=self.lr, alpha=0.95
        )

        self.scheduler = sched.ReduceLROnPlateau(
            self.optim_G, mode="min", factor=0.8, patience=10, verbose=True, min_lr=1e-6
        )
        # 初始化日志
        if self.result_path:
            os.makedirs(self.result_path, exist_ok=True)
            self.log_path = os.path.join(self.result_path, "integration_training.log")
            with open(self.log_path, "w") as f:
                f.write("=== Integration_Model Architecture ===\n")
                f.write(repr(self))
                f.write("\n\n")

    @staticmethod
    def data_processing(adatas: dict, graph_data):
        random_seed = 5000
        data_batch_size = 128
        train_ratio = 0.9

        data = AnnDataset(
            list(adatas.values()),
            [adata.uns["data_config"] for adata in adatas.values()],
            mode="train",
        )

        graph = GraphDataset(
            graph_data,
            neg_samples=10,
            weighted_sampling=True,
            deemphasize_loops=True,
        )

        graph_batch_size = ceil(graph.size / 32)
        data.getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        graph.getitem_size = max(1, round(graph_batch_size / config.DATALOADER_FETCHES_PER_BATCH))
        data_train, data_val = data.random_split(
            [train_ratio, 1 - train_ratio], random_state=random_seed
        )
        data_train.prepare_shuffle(
            num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed
        )
        data_val.prepare_shuffle(
            num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed
        )
        graph.prepare_shuffle(num_workers=config.GRAPH_SHUFFLE_NUM_WORKERS, random_seed=random_seed)
        print(f"data_train: {len(data_train)}, data_test: {len(data_val)}")

        train_loader = ParallelDataLoader(
            DataLoader(
                data_train,
                batch_size=config.DATALOADER_FETCHES_PER_BATCH,
                shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False,
            ),
            DataLoader(
                graph,
                batch_size=config.DATALOADER_FETCHES_PER_BATCH,
                shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(graph) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False,
            ),
            cycle_flags=[False, True],
        )
        val_loader = ParallelDataLoader(
            DataLoader(
                data_val,
                batch_size=config.DATALOADER_FETCHES_PER_BATCH,
                shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(data_val) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False,
            ),
            DataLoader(
                graph,
                batch_size=config.DATALOADER_FETCHES_PER_BATCH,
                shuffle=True,
                num_workers=config.DATALOADER_NUM_WORKERS,
                pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
                drop_last=len(graph) > config.DATALOADER_FETCHES_PER_BATCH,
                generator=torch.Generator().manual_seed(random_seed),
                persistent_workers=False,
            ),
            cycle_flags=[False, True],
        )
        return graph, train_loader, val_loader

    def step_discriminator_loss(self, batch: tuple):
        rna_x, atac_x, rna_xbch, atac_xbch, *_ = batch
        rna_u, rna_l = self.rna_encoder(rna_x)
        atac_u, atac_l = self.atac_encoder(atac_x)
        z_u_cat = torch.cat([rna_u.mean, atac_u.mean], dim=0).detach()
        z_flag = torch.cat(
            [
                torch.zeros(rna_x.size(0), dtype=torch.long, device=self.device),
                torch.ones(atac_x.size(0), dtype=torch.long, device=self.device),
            ],
            dim=0,
        )
        xbch_cat = torch.cat([rna_xbch, atac_xbch], dim=0)

        noise = D.Normal(0, z_u_cat.std(axis=0)).sample((z_u_cat.shape[0],))
        z_u_cat = z_u_cat + 0.5 * noise

        d_out = self.d_discriminator(z_u_cat, xbch_cat)
        d_loss = F.cross_entropy(d_out, z_flag, reduction="mean")
        return self.lam_d * d_loss

    def step_generator_loss(self, batch: tuple, use_contras: bool = True, use_disc: bool = True):
        rna_x, atac_x, rna_xbch, atac_xbch, rna_xlbl, atac_xlbl, xflag, eidx, ewt = batch
        use_contras = bool(use_contras and self.is_paired)
        rna_u, rna_l = self.rna_encoder(rna_x)
        atac_u, atac_l = self.atac_encoder(atac_x)
        rna_z = rna_u.rsample()
        atac_z = atac_u.rsample()

        # Prepare discriminator inputs
        z_u_cat = torch.cat([rna_u.mean, atac_u.mean], dim=0)
        z_flag = torch.cat(
            [
                torch.zeros(rna_x.size(0), dtype=torch.long, device=self.device),
                torch.ones(atac_x.size(0), dtype=torch.long, device=self.device),
            ],
            dim=0,
        )
        xbch_cat = torch.cat([rna_xbch, atac_xbch], dim=0)
        noise = D.Normal(0, z_u_cat.std(axis=0)).sample((z_u_cat.shape[0],))
        z_u_cat = z_u_cat + 0.5 * noise

        # Discriminator loss (subtracted from generator objective)
        if use_disc:
            d_out = self.d_discriminator(z_u_cat, xbch_cat)
            d_loss_raw = F.cross_entropy(d_out, z_flag, reduction="mean")
            d_loss_scaled = self.lam_d * d_loss_raw
        else:
            d_loss_raw = torch.tensor(0.0, device=self.device)
            d_loss_scaled = torch.tensor(0.0, device=self.device)

        # Contrastive learning
        if use_contras:
            z_p_rna = F.normalize(rna_z, dim=1)
            z_p_atac = F.normalize(atac_z, dim=1)

            sim = torch.mm(z_p_rna, z_p_atac.t())
            pos_mask = torch.zeros_like(sim, dtype=torch.bool)
            diag = torch.arange(min(sim.shape[0], sim.shape[1]), device=self.device)
            pos_mask[diag, diag] = True  # align positives by position
            contras_loss = info_nce_cross(sim, pos_mask)
            # contras_loss = info_nce_loss(rna_z, atac_z)
        else:
            contras_loss = torch.tensor(0.0, device=self.device)

        # Graph ELBO
        v = self.graph_encoder(self.eidx, self.enorm)
        g_z = v.rsample()
        g_nll = -self.graph_decoder(g_z, eidx).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, self.prior()).sum(dim=1).mean() / g_z.shape[0]
        g_elbo = (g_nll.mean()) + self.lam_kl * g_kl

        # RNA/ATAC ELBO
        rna_nll = (
            -self.rna_decoder(
                rna_z, g_z[: self.data_config["RNA"]["features"].size], rna_xbch, rna_l
            )
            .log_prob(rna_x)
            .mean()
        )
        atac_nll = (
            -self.atac_decoder(
                atac_z, g_z[self.data_config["RNA"]["features"].size :], atac_xbch, atac_l
            )
            .log_prob(atac_x)
            .mean()
        )
        rna_kld = D.kl_divergence(rna_u, self.prior()).sum(dim=1).mean() / rna_x.shape[1]
        atac_kld = D.kl_divergence(atac_u, self.prior()).sum(dim=1).mean() / atac_x.shape[1]
        rna_elbo = rna_nll + self.lam_kl * rna_kld
        atac_elbo = atac_nll + self.lam_kl * atac_kld

        vae_loss = rna_elbo + atac_elbo + self.lam_graph * g_elbo + 0.01 * contras_loss
        gen_loss = vae_loss - d_loss_scaled

        metrics = {
            "gen_loss": gen_loss,
            "vae_loss": vae_loss,
            "d_loss": d_loss_raw,  # Raw discriminator loss for logging
            "graph_ELBO_loss": g_elbo,
            "RNA_ELBO_loss": rna_elbo,
            "ATAC_ELBO_loss": atac_elbo,
            "RNA_NLL_loss": rna_nll,
            "RNA_KLD_loss": rna_kld,
            "ATAC_NLL_loss": atac_nll,
            "ATAC_KLD_loss": atac_kld,
            "graph_NLL_loss": g_nll,
            "graph_KLD_loss": g_kl,
            "contras_loss": contras_loss,
        }
        return metrics


    def train_model(self, train_loader, val_loader, epochs: int = 200, print_every=10):
        print("")
        loss_names = [
            "gen_loss",
            "vae_loss",
            "d_loss",
            "graph_ELBO_loss",
            "RNA_ELBO_loss",
            "ATAC_ELBO_loss",
            "RNA_NLL_loss",
            "RNA_KLD_loss",
            "ATAC_NLL_loss",
            "ATAC_KLD_loss",
            "graph_NLL_loss",
            "graph_KLD_loss",
            "contras_loss",
        ]
        train_history = {name: [] for name in loss_names}
        val_history = {name: [] for name in loss_names}
        best_state = None

        epochs = max(0, epochs)
        use_contras = self.is_paired and epochs > 0

        print(f"Starting Integration training on device {self.device}.")
        print(f"Epochs: {epochs}")
        if not self.is_paired:
            print("Data detected as unpaired; contrastive branch will be skipped.")

        initial_lr = self.optim_G.param_groups[0]["lr"]
        print(f"Initial learning rate: {initial_lr:.6f}")
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(f"Start training: epochs={epochs}, lr={initial_lr:.6f}\n")

        if epochs == 0:
            print("\nTraining skipped because epochs is 0.")
        else:
            print(f"\nFine-tuning for {epochs} epochs.")
            if use_contras:
                print("Contrastive loss enabled.")

            for epoch in range(epochs):
                self.train()
                train_metrics = {name: [] for name in loss_names}
                for batch in train_loader:
                    batch = [b.to(self.device) for b in batch]

                    self.optim_D.zero_grad()
                    d_loss = self.step_discriminator_loss(batch)
                    d_loss.backward()
                    self.optim_D.step()

                    self.optim_G.zero_grad()
                    metrics = self.step_generator_loss(
                        batch, use_contras=use_contras, use_disc=True
                    )
                    metrics["gen_loss"].backward()
                    self.optim_G.step()

                    for name, val in metrics.items():
                        train_metrics[name].append(val.detach().cpu().item())

                self.eval()
                val_metrics = {name: [] for name in loss_names}
                with torch.no_grad():
                    for batch in val_loader:
                        batch = [b.to(self.device) for b in batch]
                        metrics_val = self.step_generator_loss(
                            batch, use_contras=use_contras, use_disc=True
                        )
                        for name, val in metrics_val.items():
                            val_metrics[name].append(val.detach().cpu().item())

                val_vae = np.mean(val_metrics["vae_loss"])

                for name in loss_names:
                    train_history[name].append(np.mean(train_metrics[name]))
                    val_history[name].append(np.mean(val_metrics[name]))

                old_lr = self.optim_G.param_groups[0]["lr"]
                self.scheduler.step(val_vae)
                new_lr = self.optim_G.param_groups[0]["lr"]
                if new_lr != old_lr:
                    print(
                        f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f} at epoch {epoch + 1} "
                        f"(val_vae_loss={val_vae:.4f})"
                    )
                    for pg in self.optim_D.param_groups:
                        pg["lr"] = new_lr
                if self.log_path:
                    with open(self.log_path, "a") as f:
                        log_line = (
                            f"Epoch {epoch + 1}/{epochs} | "
                            + " | ".join(
                                [
                                    f"train_{name}={np.mean(train_metrics[name]):.4f}"
                                    for name in loss_names
                                ]
                            )
                            + f" | lr={self.optim_G.param_groups[0]['lr']:.6f}\n"
                            + " | ".join(
                                [
                                    f"val_{name}={np.mean(val_metrics[name]):.4f}"
                                    for name in loss_names
                                ]
                            )
                            + "\n"
                        )
                        f.write(log_line)

                if val_vae < self.best_val_vae_loss - 1e-8:
                    self.best_val_vae_loss = val_vae
                    self.no_improve_count = 0
                    best_state = {k: v.cpu() for k, v in self.state_dict().items()}
                else:
                    self.no_improve_count += 1
                    if self.no_improve_count >= self.early_stop_patience:
                        print(f"Early stopping at epoch {epoch + 1} (no improvement in vae_loss)")
                        break

                if ((epoch + 1) % print_every == 0) or epoch == 0:
                    print(f"\nEpoch {epoch + 1}/{epochs}")
                    print(
                        "Train: "
                        + ",  ".join(
                            [f"{name}: {np.mean(train_metrics[name]):.4f}" for name in loss_names]
                        )
                    )
                    print(
                        "Val:   "
                        + ",  ".join(
                            [f"{name}: {np.mean(val_metrics[name]):.4f}" for name in loss_names]
                        )
                    )
                    print(f"Best val vae_loss so far: {self.best_val_vae_loss:.4f}")

        # Save best and final checkpoints
        if best_state and self.result_path:
            os.makedirs(self.result_path, exist_ok=True)
            best_path = os.path.join(self.result_path, "Integration_best_model.pt")
            torch.save(best_state, best_path)
            print(f"Best validation-model saved at {best_path}")

        if self.result_path:
            final_path = os.path.join(self.result_path, "Integration_final_model.pt")
            torch.save(self.state_dict(), final_path)
            print(f"Final model saved at {final_path}")

    @torch.no_grad()
    def encode_data(self, encoder, adata, batch_size=128):
        """
        Generic encoder for RNA or ATAC modalities.
        encoder: self.rna_encoder or self.atac_encoder
        adata: AnnData object to encode
        """
        self.eval()
        data = AnnDataset(
            [adata], [adata.uns["data_config"]], mode="train", getitem_size=batch_size
        )
        data_loader = DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=False,
            persistent_workers=False,
        )
        result = []
        for batch in data_loader:
            x = batch[0].to(self.device)
            u, l = encoder(x)
            result.append(u.mean.detach().cpu())
        return torch.cat(result, dim=0).numpy()

    @torch.no_grad()
    def encode_feature(self, n_sample: int | None = None) -> np.ndarray:
        """
        Compute graph (feature/vertex) embedding

        Parameters
        ----------
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        graph_embedding
            Graph (feature) embedding with shape:
            - (n_vertices * latent_dim) if n_sample is None
            - (n_vertices * n_sample * latent_dim) if n_sample is not None
        """
        self.eval()

        # Encode graph vertices using stored graph structure
        v = self.graph_encoder(self.eidx, self.enorm)

        if n_sample:
            # Sample n_sample times from the distribution
            samples = torch.cat([v.sample((1,)).cpu() for _ in range(n_sample)])
            return samples.permute(1, 0, 2).numpy()

        # Return the mean of the distribution
        return v.mean.detach().cpu().numpy()
