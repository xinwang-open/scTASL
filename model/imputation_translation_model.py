from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn as nn
import torch.optim.lr_scheduler as sched

from utils.common.config import config
from utils.common.dataloader import AnnDataset, DataLoader
from utils.common.losses import info_nce_loss

from .modules import Decoder, Encoder, NaiveAffineTransform, Prior

BatchType = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]  # rna_x, atac_x, rna_xbch, atac_xbch


@dataclass(frozen=True)
class TrainingStageConfig:
    warmup_rna_epochs: int = 50
    warmup_atac_epochs: int = 50
    map_epochs: int = 200
    print_every: int = 10
    save_best: bool = True


class ImputationTranslationModel(nn.Module):
    """
    Imputation/Translation model:
      1) Warm-up RNA-VAE and ATAC-VAE separately
      2) Train NaiveAffineTransform to learn cross-modality mapping in latent space

    Mapping stage objectives:
      - supervised translation (data space)
      - cycle consistency (data space)
      - latent alignment (paired-friendly, not MMD)
      - unimodal anchors to stabilize VAEs while mapping
    """

    def __init__(
        self,
        data_config: dict[str, Any],
        emb_size: int = 256,
        latent_dim: int = 32,
        dropout_rate: float = 0.2,
        lr_vae: float = 2e-3,
        lr_map: float = 2e-3,
        lam_kl: float = 0.3,
        # mapping-stage weights
        lam_trans: float = 1.0,
        lam_cycle: float = 1.0,
        lam_nce: float = 0.1,
        lam_self: float = 0.5,
        lam_encoder_kl: float = 0.15,
        lam_map_kl: float = 0.1,
        # NaiveAffineTransform
        affine_num: int = 8,
        result_path: str | None = None,
        early_stop_patience: int = 100,
        fine_tune_vae_in_mapping: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.data_config = self._normalize_data_config(data_config)

        # compatibility flag
        self.is_paired = bool(self.data_config.get("is_paired", True))

        # hyperparams
        self.lr_vae = float(lr_vae)
        self.lr_map = float(lr_map)
        self.lam_kl = float(lam_kl)

        # mapping-stage weights
        self.lam_trans = float(lam_trans)
        self.lam_cycle = float(lam_cycle)
        self.lam_nce = float(lam_nce)
        self.lam_self = float(lam_self)
        self.lam_encoder_kl = float(lam_encoder_kl)
        self.lam_map_kl = float(lam_map_kl)

        self.affine_num = int(affine_num)
        self.fine_tune_vae_in_mapping = bool(fine_tune_vae_in_mapping)

        # early stop
        self.early_stop_patience = int(early_stop_patience)
        self.no_improve_count = 0
        self.best_val_loss = float("inf")

        # paths
        self.result_path = result_path
        self.log_path: str | None = None

        # modules
        self.rna_encoder = Encoder(
            input_dim=len(self.data_config["RNA"]["features"]),
            emb_size=emb_size,
            output_dim=latent_dim,
            dropout_rate=dropout_rate,
            use_library_norm=False,
        ).to(self.device)

        self.atac_encoder = Encoder(
            input_dim=len(self.data_config["ATAC"]["features"]),
            emb_size=emb_size,
            output_dim=latent_dim,
            dropout_rate=dropout_rate,
            use_library_norm=False,
        ).to(self.device)

        self.rna_decoder = Decoder(
            input_dim=latent_dim,
            emb_size=emb_size,
            output_dim=len(self.data_config["RNA"]["features"]),
            n_batches=max(1, self.data_config["RNA"]["batches"].size),
            use_library_norm=False,
        ).to(self.device)

        self.atac_decoder = Decoder(
            input_dim=latent_dim,
            emb_size=emb_size,
            output_dim=len(self.data_config["ATAC"]["features"]),
            n_batches=max(1, self.data_config["ATAC"]["batches"].size),
            use_library_norm=False,
        ).to(self.device)

        self.prior = Prior().to(self.device)

        self.r2a_mapper = NaiveAffineTransform(
            input_dim=latent_dim,
            z_dim=latent_dim,
            affine_num=self.affine_num,
            reverse=False,
        ).to(self.device)

        self.a2r_mapper = NaiveAffineTransform(
            input_dim=latent_dim,
            z_dim=latent_dim,
            affine_num=self.affine_num,
            reverse=True,
        ).to(self.device)

        # optimizers / schedulers
        self.optim_rna = torch.optim.AdamW(
            itertools.chain(self.rna_encoder.parameters(), self.rna_decoder.parameters()),
            lr=self.lr_vae,
            weight_decay=1e-4,
        )
        self.optim_atac = torch.optim.AdamW(
            itertools.chain(self.atac_encoder.parameters(), self.atac_decoder.parameters()),
            lr=self.lr_vae,
            weight_decay=1e-4,
        )

        map_params: list[nn.Parameter] = list(self.r2a_mapper.parameters()) + list(
            self.a2r_mapper.parameters()
        )
        if self.fine_tune_vae_in_mapping:
            map_params += list(self.rna_encoder.parameters()) + list(self.atac_encoder.parameters())
            map_params += list(self.rna_decoder.parameters()) + list(self.atac_decoder.parameters())

        self.optim_map = torch.optim.AdamW(map_params, lr=self.lr_map, weight_decay=1e-4)

        self.sched_rna = self._build_plateau_scheduler(self.optim_rna)
        self.sched_atac = self._build_plateau_scheduler(self.optim_atac)
        self.sched_map = self._build_plateau_scheduler(self.optim_map)

        self._init_logging()

    # ---------------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _build_plateau_scheduler(optimizer: torch.optim.Optimizer) -> sched.ReduceLROnPlateau:
        return sched.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-6)

    @staticmethod
    def _normalize_data_config(data_config: dict[str, Any]) -> dict[str, Any]:
        cfg = dict(data_config)  # shallow copy

        for mod in ("RNA", "ATAC"):
            if cfg[mod].get("batches", None) is None:
                cfg[mod]["batches"] = pd.Index([])
            else:
                cfg[mod]["batches"] = pd.Index(cfg[mod]["batches"])

            cfg[mod]["features"] = pd.Index(cfg[mod]["features"])

        return cfg

    def _init_logging(self) -> None:
        if not self.result_path:
            return
        os.makedirs(self.result_path, exist_ok=True)
        self.log_path = os.path.join(self.result_path, "training.log")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=== ImputationTranslationModel (2xVAE + 2xNaiveAffineTransform) ===\n")
            f.write(repr(self))
            f.write("\n\n")

    def _log(self, msg: str) -> None:
        print(msg)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    @staticmethod
    def _move_batch_to_device(
        batch: list[torch.Tensor], device: torch.device
    ) -> list[torch.Tensor]:
        return [b.to(device) for b in batch]

    @staticmethod
    def _nll(dist: Any, x: torch.Tensor) -> torch.Tensor:
        # dist.log_prob(x): [B, ...] -> scalar mean
        return (-dist.log_prob(x)).mean()

    @staticmethod
    def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=1, keepdim=True) + eps)

    def _latent_kl_to_prior(self, z: torch.Tensor) -> torch.Tensor:
        """
        Treat z as mean of Normal(z, 1), compute KL to model prior.
        Normalize by feature dim for stable scale.
        """
        prior = self.prior()
        dist = D.Normal(z, torch.ones_like(z))
        return D.kl_divergence(dist, prior).sum(dim=1).mean() / z.shape[1]

    # ---------------------------------------------------------------------
    # Data loader helper
    # ---------------------------------------------------------------------
    @staticmethod
    def data_processing(adatas: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
        random_seed = 5000
        data_batch_size = 128
        train_ratio = 0.9

        data = AnnDataset(
            list(adatas.values()),
            [adata.uns["data_config"] for adata in adatas.values()],
            mode="train",
        )
        data.getitem_size = max(1, round(data_batch_size / config.DATALOADER_FETCHES_PER_BATCH))

        data_train, data_val = data.random_split(
            [train_ratio, 1 - train_ratio], random_state=random_seed
        )
        data_train.prepare_shuffle(
            num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed
        )
        data_val.prepare_shuffle(
            num_workers=config.ARRAY_SHUFFLE_NUM_WORKERS, random_seed=random_seed
        )

        print(f"data_train: {len(data_train)}, data_val: {len(data_val)}")

        train_loader = DataLoader(
            data_train,
            batch_size=config.DATALOADER_FETCHES_PER_BATCH,
            shuffle=True,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=len(data_train) > config.DATALOADER_FETCHES_PER_BATCH,
            generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=False,
        )

        val_loader = DataLoader(
            data_val,
            batch_size=config.DATALOADER_FETCHES_PER_BATCH,
            shuffle=True,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=len(data_val) > config.DATALOADER_FETCHES_PER_BATCH,
            generator=torch.Generator().manual_seed(random_seed),
            persistent_workers=False,
        )
        return train_loader, val_loader

    # ---------------------------------------------------------------------
    # Warm-up (unimodal VAE)
    # ---------------------------------------------------------------------
    def _step_unimodal_vae(
        self, x: torch.Tensor, xbch: torch.Tensor, modality: Literal["RNA", "ATAC"]
    ):
        prior = self.prior()

        if modality == "RNA":
            dist, _, _ = self.rna_encoder(x, return_hidden=True)
            z = dist.rsample()
            nll = -self.rna_decoder(z, xbch).log_prob(x).mean()
        elif modality == "ATAC":
            dist, _, _ = self.atac_encoder(x, return_hidden=True)
            z = dist.rsample()
            nll = -self.atac_decoder(z, xbch).log_prob(x).mean()
        else:
            raise ValueError("modality must be 'RNA' or 'ATAC'")

        kld = D.kl_divergence(dist, prior).sum(dim=1).mean() / x.shape[1]
        elbo = nll + self.lam_kl * kld
        return elbo, nll, kld

    def _run_warmup(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        modality: Literal["RNA", "ATAC"],
        epochs: int,
        print_every: int,
    ) -> None:
        opt = self.optim_rna if modality == "RNA" else self.optim_atac
        sch = self.sched_rna if modality == "RNA" else self.sched_atac

        for ep in range(int(epochs)):
            self.train()
            tr_vals: list[float] = []

            for batch in train_loader:
                batch = self._move_batch_to_device(batch, self.device)
                rna_x, atac_x, rna_xbch, atac_xbch = batch[0], batch[1], batch[2], batch[3]

                opt.zero_grad(set_to_none=True)
                if modality == "RNA":
                    elbo, _, _ = self._step_unimodal_vae(rna_x, rna_xbch, "RNA")
                else:
                    elbo, _, _ = self._step_unimodal_vae(atac_x, atac_xbch, "ATAC")

                elbo.backward()
                opt.step()
                tr_vals.append(float(elbo.detach().cpu().item()))

            self.eval()
            va_vals: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = self._move_batch_to_device(batch, self.device)
                    rna_x, atac_x, rna_xbch, atac_xbch = batch[0], batch[1], batch[2], batch[3]
                    if modality == "RNA":
                        elbo, _, _ = self._step_unimodal_vae(rna_x, rna_xbch, "RNA")
                    else:
                        elbo, _, _ = self._step_unimodal_vae(atac_x, atac_xbch, "ATAC")
                    va_vals.append(float(elbo.detach().cpu().item()))

            tr = float(np.mean(tr_vals)) if tr_vals else float("nan")
            va = float(np.mean(va_vals)) if va_vals else float("nan")
            sch.step(va)

            if (ep == 0) or ((ep + 1) % int(print_every) == 0):
                self._log(
                    f"[Warmup-{modality}] Epoch {ep+1}/{epochs} | train_elbo={tr:.4f} | val_elbo={va:.4f}"
                )

    # ---------------------------------------------------------------------
    # Mapping step
    # ---------------------------------------------------------------------
    def _step_mapping(self, batch: BatchType) -> dict[str, torch.Tensor]:
        rna_x, atac_x, rna_xbch, atac_xbch = batch

        prior = self.prior()

        # encode -> latent mean
        rna_dist, _, _ = self.rna_encoder(rna_x, return_hidden=True)
        atac_dist, _, _ = self.atac_encoder(atac_x, return_hidden=True)
        z_r = rna_dist.mean
        z_a = atac_dist.mean

        # self reconstruction anchor
        rna_self_nll = self._nll(self.rna_decoder(z_r, rna_xbch), rna_x)
        atac_self_nll = self._nll(self.atac_decoder(z_a, atac_xbch), atac_x)
        self_loss = rna_self_nll + atac_self_nll

        # encoder KL
        rna_kld = D.kl_divergence(rna_dist, prior).sum(dim=1).mean() / rna_x.shape[1]
        atac_kld = D.kl_divergence(atac_dist, prior).sum(dim=1).mean() / atac_x.shape[1]
        encoder_kld = rna_kld + atac_kld

        # latent mapping
        z_r2a = self.r2a_mapper(z_r)
        z_a2r = self.a2r_mapper(z_a)

        # weak mapped latent regularization
        map_latent_kld = self._latent_kl_to_prior(z_r2a) + self._latent_kl_to_prior(z_a2r)

        # supervised translation
        atac_dist_from_rna = self.atac_decoder(z_r2a, atac_xbch)
        rna_dist_from_atac = self.rna_decoder(z_a2r, rna_xbch)

        atac_nll_from_rna = self._nll(atac_dist_from_rna, atac_x)
        rna_nll_from_atac = self._nll(rna_dist_from_atac, rna_x)

        trans_loss = atac_nll_from_rna + rna_nll_from_atac

        # cycle reconstruction
        z_r_cycle = self.a2r_mapper(z_r2a)
        z_a_cycle = self.r2a_mapper(z_a2r)

        rna_cycle_nll = self._nll(self.rna_decoder(z_r_cycle, rna_xbch), rna_x)
        atac_cycle_nll = self._nll(self.atac_decoder(z_a_cycle, atac_xbch), atac_x)

        cycle_loss = rna_cycle_nll + atac_cycle_nll

        # direct contrastive alignment
        nce_loss = info_nce_loss(z_a, z_r2a) + info_nce_loss(z_r, z_a2r)

        # total mapping loss
        loss = (
            self.lam_trans * trans_loss
            + self.lam_cycle * cycle_loss
            + self.lam_nce * nce_loss
            + self.lam_self * self_loss
            + self.lam_encoder_kl * encoder_kld
            + self.lam_map_kl * map_latent_kld
        )

        return {
            "map_loss": loss,
            # main losses
            "trans_loss": trans_loss,
            "cycle_loss": cycle_loss,
            "nce_loss": nce_loss,
            "self_loss": self_loss,
            "encoder_kld": encoder_kld,
            "map_latent_kld": map_latent_kld,
            # components
            "rna_self_nll": rna_self_nll,
            "atac_self_nll": atac_self_nll,
            "rna_kld": rna_kld,
            "atac_kld": atac_kld,
            "atac_nll_from_rna": atac_nll_from_rna,
            "rna_nll_from_atac": rna_nll_from_atac,
            "rna_cycle_nll": rna_cycle_nll,
            "atac_cycle_nll": atac_cycle_nll,
        }

    def _run_mapping(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        print_every: int,
        save_best: bool,
    ) -> None:
        self.best_val_loss = float("inf")
        self.no_improve_count = 0

        log_keys = [
            "trans_loss",
            "cycle_loss",
            "nce_loss",
            "self_loss",
            "encoder_kld",
            "rna_kld",
            "atac_kld",
            "map_latent_kld",
        ]

        for ep in range(int(epochs)):
            # train
            self.train()
            tr_vals: list[float] = []
            tr_sums = {k: 0.0 for k in log_keys}
            tr_count = 0

            for batch in train_loader:
                batch = self._move_batch_to_device(batch, self.device)
                rna_x, atac_x, rna_xbch, atac_xbch = batch[0], batch[1], batch[2], batch[3]

                self.optim_map.zero_grad(set_to_none=True)
                metrics = self._step_mapping((rna_x, atac_x, rna_xbch, atac_xbch))
                metrics["map_loss"].backward()
                self.optim_map.step()

                tr_vals.append(float(metrics["map_loss"].detach().cpu().item()))
                for k in log_keys:
                    tr_sums[k] += float(metrics[k].detach().cpu().item())
                tr_count += 1

            # val
            self.eval()
            va_vals: list[float] = []
            va_sums = {k: 0.0 for k in log_keys}
            va_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = self._move_batch_to_device(batch, self.device)
                    rna_x, atac_x, rna_xbch, atac_xbch = batch[0], batch[1], batch[2], batch[3]

                    metrics = self._step_mapping((rna_x, atac_x, rna_xbch, atac_xbch))
                    va_vals.append(float(metrics["map_loss"].detach().cpu().item()))
                    for k in log_keys:
                        va_sums[k] += float(metrics[k].detach().cpu().item())
                    va_count += 1

            tr = float(np.mean(tr_vals)) if tr_vals else float("nan")
            va = float(np.mean(va_vals)) if va_vals else float("nan")
            tr_avg = {k: tr_sums[k] / max(tr_count, 1) for k in log_keys}
            va_avg = {k: va_sums[k] / max(va_count, 1) for k in log_keys}

            self.sched_map.step(va)

            improved = va < self.best_val_loss - 1e-8
            if improved:
                self.best_val_loss = va
                self.no_improve_count = 0
                if save_best and self.result_path:
                    best_path = os.path.join(self.result_path, "Translation_best_model.pt")
                    torch.save(self.state_dict(), best_path)
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.early_stop_patience:
                    self._log(f"[Mapping] Early stopping at epoch {ep+1} (no improve)")
                    break

            if (ep == 0) or ((ep + 1) % int(print_every) == 0):
                self._log(
                    f"[Mapping] Epoch {ep+1}/{epochs} | train_map_loss={tr:.4f} | val_map_loss={va:.4f}"
                )
                self._log(
                    "[Mapping-Train] " + " | ".join([f"{k}={tr_avg[k]:.4f}" for k in log_keys])
                )
                self._log("[Mapping-Val] " + " | ".join([f"{k}={va_avg[k]:.4f}" for k in log_keys]))

    # ---------------------------------------------------------------------
    # Public training API
    # ---------------------------------------------------------------------
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        warmup_rna_epochs: int = 50,
        warmup_atac_epochs: int = 50,
        map_epochs: int = 200,
        print_every: int = 10,
        save_best: bool = True,
    ) -> None:
        self._log(f"Device={self.device}")

        self._log(f"StageA: warmup RNA epochs={warmup_rna_epochs}")
        self._run_warmup(train_loader, val_loader, "RNA", warmup_rna_epochs, print_every)

        self._log(f"StageB: warmup ATAC epochs={warmup_atac_epochs}")
        self._run_warmup(train_loader, val_loader, "ATAC", warmup_atac_epochs, print_every)

        self._log(f"StageC: mapping epochs={map_epochs}")
        self._run_mapping(train_loader, val_loader, map_epochs, print_every, save_best)

        if self.result_path:
            final_path = os.path.join(self.result_path, "Translation_final_model.pt")
            torch.save(self.state_dict(), final_path)
            self._log(f"Final translation model saved at {final_path}")

    @torch.no_grad()
    def impute_adata(
        self,
        adata: Any,
        modality: Literal["RNA", "ATAC"] = "RNA",
        batch_size: int = 128,
        use_mean_latent: bool = True,
        return_numpy: bool = True,
    ):
        """
        Same-modality reconstruction via the unimodal VAE.
        Encode the input -> sample/mean in latent space -> decode back.
        This serves as imputation: noisy/dropout-corrupted entries are
        smoothed by the learned data manifold.

        Parameters
        ----------
        adata      : AnnData with `uns["data_config"]` set.
        modality   : "RNA" or "ATAC" — which VAE to use.
        batch_size : inference batch size.
        use_mean_latent : if True use posterior mean (deterministic);
                        if False draw one sample (stochastic imputation).
        return_numpy    : return np.ndarray if True, else torch.Tensor.
        """
        self.eval()
        if modality not in ("RNA", "ATAC"):
            raise ValueError("modality must be 'RNA' or 'ATAC'")

        data = AnnDataset(
            [adata],
            [adata.uns["data_config"]],
            mode="train",
            getitem_size=batch_size,
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

        outputs: list[torch.Tensor] = []

        for batch in data_loader:
            x = batch[0].to(self.device)
            xbch = batch[2].to(self.device)

            if modality == "RNA":
                dist, _, _ = self.rna_encoder(x, return_hidden=True)
                z = dist.mean if use_mean_latent else dist.rsample()
                recon_dist = self.rna_decoder(z, xbch, l=None)
            else:
                dist, _, _ = self.atac_encoder(x, return_hidden=True)
                z = dist.mean if use_mean_latent else dist.rsample()
                recon_dist = self.atac_decoder(z, xbch, l=None)

            outputs.append(recon_dist.mean.detach().cpu())

        imputed = torch.cat(outputs, dim=0)
        return imputed.numpy() if return_numpy else imputed

    # ---------------------------------------------------------------------
    # Translation (AnnData)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def translate_adata(
        self,
        adata: Any,
        direction: Literal["RNA2ATAC", "ATAC2RNA"] = "RNA2ATAC",
        batch_size: int = 128,
        use_mean_latent: bool = True,
        return_numpy: bool = True,
    ):
        self.eval()
        if direction not in ("RNA2ATAC", "ATAC2RNA"):
            raise ValueError("direction must be 'RNA2ATAC' or 'ATAC2RNA'")

        data = AnnDataset(
            [adata],
            [adata.uns["data_config"]],
            mode="train",
            getitem_size=batch_size,
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

        outputs: list[torch.Tensor] = []

        for batch in data_loader:
            # keep consistent with training:
            x = batch[0].to(self.device)
            xbch = batch[2].to(self.device)

            if direction == "RNA2ATAC":
                # encode RNA -> z_r
                rna_dist, _, _ = self.rna_encoder(x, return_hidden=True)
                z_r = rna_dist.mean if use_mean_latent else rna_dist.rsample()

                # map z_r -> z_r2a
                z_r2a = self.r2a_mapper(z_r)

                # decode ATAC with atac_xbch (IMPORTANT)
                atac_dist_from_rna = self.atac_decoder(z_r2a, xbch, l=None)
                outputs.append(atac_dist_from_rna.mean.detach().cpu())

            else:  # ATAC2RNA
                atac_dist, _, _ = self.atac_encoder(x, return_hidden=True)
                z_a = atac_dist.mean if use_mean_latent else atac_dist.rsample()

                z_a2r = self.a2r_mapper(z_a)

                # decode RNA with rna_xbch (IMPORTANT)
                rna_dist_from_atac = self.rna_decoder(z_a2r, xbch, l=None)
                outputs.append(rna_dist_from_atac.mean.detach().cpu())

        translated = torch.cat(outputs, dim=0)
        return translated.numpy() if return_numpy else translated

