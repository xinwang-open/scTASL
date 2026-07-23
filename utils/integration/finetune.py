from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import torch

from model.integration_model import Integration_Model
from utils.common.data_configure import configure_dataset


def _prepare_modality(adata: ad.AnnData, domain_name: str) -> ad.AnnData:
    out = adata.copy()
    out.obs["domain"] = pd.Categorical([domain_name] * out.n_obs)
    if "counts" in out.layers:
        out.X = out.layers["counts"]
        out.uns.pop("log1p", None)
    return out


def subset_target_cell_type(
    rna: ad.AnnData,
    atac: ad.AnnData,
    target_cell_type: str,
    cell_type_key: str = "cell_type",
) -> dict[str, ad.AnnData]:
    if cell_type_key not in rna.obs:
        raise KeyError(f"`{cell_type_key}` not found in RNA obs.")
    if cell_type_key not in atac.obs:
        raise KeyError(f"`{cell_type_key}` not found in ATAC obs.")

    rna_sub = rna[rna.obs[cell_type_key] == target_cell_type].copy()
    atac_sub = atac[atac.obs[cell_type_key] == target_cell_type].copy()

    if rna_sub.n_obs == 0:
        raise ValueError(f"No RNA cells found for cell type: {target_cell_type}")
    if atac_sub.n_obs == 0:
        raise ValueError(f"No ATAC cells found for cell type: {target_cell_type}")

    rna_sub = _prepare_modality(rna_sub, "scRNA-seq")
    atac_sub = _prepare_modality(atac_sub, "scATAC-seq")

    return {"RNA": rna_sub, "ATAC": atac_sub}


def _configure_dataset_safely(adata: ad.AnnData, use_obs_names: bool = True) -> None:
    # Avoid configure_dataset warning branch that relies on configure_dataset.logger.
    if "data_config" in adata.uns:
        del adata.uns["data_config"]
    configure_dataset(adata, use_obs_names=use_obs_names)


def _freeze_graph_branch(model: Integration_Model, freeze: bool = True) -> None:
    for p in model.graph_encoder.parameters():
        p.requires_grad = not freeze
    for p in model.graph_decoder.parameters():
        p.requires_grad = not freeze


def _reset_early_stop_state(model: Integration_Model) -> None:
    model.best_val_vae_loss = float("inf")
    model.no_improve_count = 0


def _encode_and_attach(model: Integration_Model, adatas: dict[str, ad.AnnData]) -> None:
    adatas["RNA"].obsm["embedding"] = model.encode_data(model.rna_encoder, adatas["RNA"])
    adatas["ATAC"].obsm["embedding"] = model.encode_data(model.atac_encoder, adatas["ATAC"])

    feature_embeddings = model.encode_feature()
    n_rna_features = len(model.data_config["RNA"]["features"])
    adatas["RNA"].varm["feature_embedding"] = feature_embeddings[:n_rna_features]
    adatas["ATAC"].varm["feature_embedding"] = feature_embeddings[n_rna_features:]


def _compute_cell_type_prototypes(adatas: dict[str, ad.AnnData]) -> dict[str, np.ndarray]:
    rna_proto = adatas["RNA"].obsm["embedding"].mean(axis=0)
    atac_proto = adatas["ATAC"].obsm["embedding"].mean(axis=0)
    return {
        "RNA": rna_proto,
        "ATAC": atac_proto,
    }


def _save_finetuned_adatas(
    adatas: dict[str, ad.AnnData],
    output_path: Path,
    target_cell_type: str,
    cell_type_key: str,
) -> dict[str, str]:
    safe_cell_type = str(target_cell_type).replace(" ", "_").replace("/", "_")
    adata_dir = output_path / "adata"
    adata_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "RNA": adata_dir / f"rna_finetuned_{safe_cell_type}.h5ad",
        "ATAC": adata_dir / f"atac_finetuned_{safe_cell_type}.h5ad",
    }
    for mod in ["RNA", "ATAC"]:
        if cell_type_key not in adatas[mod].obs:
            raise KeyError(f"`{cell_type_key}` not found in adatas['{mod}'].obs")
        if "data_config" in adatas[mod].uns:
            del adatas[mod].uns["data_config"]
        adatas[mod].write(str(paths[mod]), compression="gzip")
    return {k: str(v) for k, v in paths.items()}


def finetune_single_cell_type(
    rna: ad.AnnData,
    atac: ad.AnnData,
    graph_data: Any,
    base_ckpt_path: str,
    output_dir: str,
    target_cell_type: str,
    cell_type_key: str = "cell_type",
    use_obs_names: bool = True,
    is_paired: bool = True,
    lr: float = 2e-4,
    stage1_epochs: int = 20,
    stage2_epochs: int = 0,
    freeze_graph_stage1: bool = True,
    print_every: int = 10,
    save_h5ad: bool = True,
) -> dict[str, Any]:
    """
    Fine-tune Integration model on a single target cell type.

    Stage 1 (default): freeze graph branch and update encoders/decoders.
    Stage 2 (optional): unfreeze graph branch for feature_embedding adaptation.
    """
    adatas = subset_target_cell_type(
        rna=rna,
        atac=atac,
        target_cell_type=target_cell_type,
        cell_type_key=cell_type_key,
    )

    for adata0 in adatas.values():
        _configure_dataset_safely(adata0, use_obs_names=use_obs_names)

    data_config = {name: adata0.uns["data_config"] for name, adata0 in adatas.items()}
    vertices = data_config["RNA"]["features"] + data_config["ATAC"]["features"]

    graph, train_loader, val_loader = Integration_Model.data_processing(
        adatas=adatas,
        graph_data=graph_data,
    )

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    model = Integration_Model(
        data_config=data_config,
        graph=graph,
        vertices=vertices,
        result_path=str(output_path),
        is_paired=is_paired,
        lr=lr,
    )

    state_dict = torch.load(base_ckpt_path, map_location=model.device)
    model.load_state_dict(state_dict, strict=False)

    if stage1_epochs > 0:
        _freeze_graph_branch(model, freeze=freeze_graph_stage1)
        _reset_early_stop_state(model)
        model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=stage1_epochs,
            print_every=print_every,
        )

    if stage2_epochs > 0:
        _freeze_graph_branch(model, freeze=False)
        _reset_early_stop_state(model)
        model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=stage2_epochs,
            print_every=print_every,
        )

    _encode_and_attach(model, adatas)
    prototypes = _compute_cell_type_prototypes(adatas)
    saved_h5ad_paths = {}
    if save_h5ad:
        saved_h5ad_paths = _save_finetuned_adatas(
            adatas=adatas,
            output_path=output_path,
            target_cell_type=target_cell_type,
            cell_type_key=cell_type_key,
        )

    return {
        "model": model,
        "adatas": adatas,
        "prototypes": prototypes,
        "n_cells": {
            "RNA": int(adatas["RNA"].n_obs),
            "ATAC": int(adatas["ATAC"].n_obs),
        },
        "output_dir": str(output_path),
        "saved_h5ad_paths": saved_h5ad_paths,
    }
