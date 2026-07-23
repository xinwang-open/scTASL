import pickle
from pathlib import Path

import anndata
import pandas as pd
from anndata import AnnData

# from .utils import config, logged


def configure_dataset(
    adata: AnnData,
    use_batch: str | None = None,
    use_cell_type: str | None = None,
    use_obs_names: bool = False,
) -> None:
    r"""
    Configure dataset for model training

    Parameters
    ----------
    adata
        Dataset to be configured
    use_batch
        Data batch to use (key in ``adata.obs``)
    use_cell_type
        Data cell type to use (key in ``adata.obs``)
    use_obs_names
        Whether to use ``obs_names`` to mark paired cells across
        different datasets

    Note
    -----
    The ``use_rep`` option applies to encoder inputs, but not the decoders,
    which are always fitted on data in the original space.
    """
    if "data_config" in adata.uns:
        configure_dataset.logger.warning(
            "`configure_dataset` has already been called. "
            "Previous configuration will be overwritten!"
        )
    data_config = {}
    if use_batch:
        if use_batch not in adata.obs:
            raise ValueError("Invalid `use_batch`!")
        data_config["use_batch"] = use_batch
        data_config["batches"] = (
            pd.Index(adata.obs[use_batch]).dropna().drop_duplicates().sort_values().to_numpy()
        )  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_batch"] = None
        data_config["batches"] = None

    if use_cell_type:
        if use_cell_type not in adata.obs:
            raise ValueError("Invalid `use_cell_type`!")
        data_config["use_cell_type"] = use_cell_type
        data_config["cell_types"] = (
            pd.Index(adata.obs[use_cell_type]).dropna().drop_duplicates().sort_values().to_numpy()
        )  # AnnData does not support saving pd.Index in uns
    else:
        data_config["use_cell_type"] = None
        data_config["cell_types"] = None

    data_config["use_obs_names"] = use_obs_names
    data_config["features"] = adata.var_names.to_numpy().tolist()
    adata.uns["data_config"] = data_config


def load_omics_inputs(
    *,
    rna_path: str | Path,
    atac_path: str | Path,
    graph_path: str | Path | None = None,
    additional_required_paths: dict[str, str | Path] | None = None,
    backed: str | None = None,
    require_counts: bool = True,
    require_cell_type: bool = False,
    cell_type_key: str = "cell_type",
    require_paired: bool = True,
    paired_task: str = "paired analysis",
    close_backed: bool = False,
) -> tuple[AnnData, AnnData, object | None, pd.DataFrame]:
    """Load RNA/ATAC AnnData inputs from explicit source paths.

    Parameters
    ----------
    rna_path
        RNA AnnData source path.
    atac_path
        ATAC AnnData source path.
    graph_path
        Optional graph pickle path to check and load.
    additional_required_paths
        Extra files that must exist but are not loaded here, such as a
        checkpoint.
    backed
        Optional AnnData backed mode, for example ``"r"``.
    require_counts
        Whether each AnnData object must contain ``layers["counts"]``.
    require_cell_type
        Whether each AnnData object must contain the cell-type annotation.
        Cell-type labels are usually not required for model training, but
        tutorial notebooks may require them for supervised evaluation and
        labeled UMAP figures.
    cell_type_key
        Cell-type annotation key expected in ``obs``.
    require_paired
        Whether RNA and ATAC observation names must match in the same order.
    paired_task
        Short task name used in the paired-cell error message.
    close_backed
        Close backed AnnData files before returning. Use this only when the
        caller needs the summary but will not use the returned AnnData objects.
    """
    input_paths = {"RNA": Path(rna_path), "ATAC": Path(atac_path)}
    if graph_path is not None:
        input_paths["graph"] = Path(graph_path)
    if additional_required_paths:
        input_paths.update(
            {name: Path(path) for name, path in additional_required_paths.items()}
        )

    missing_files = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required input file(s):\n- " + "\n- ".join(missing_files)
        )

    rna = anndata.read_h5ad(input_paths["RNA"], backed=backed)
    atac = anndata.read_h5ad(input_paths["ATAC"], backed=backed)
    graph_data = None
    if graph_path is not None:
        with input_paths["graph"].open("rb") as handle:
            graph_data = pickle.load(handle)

    for omics_name, adata_obj in {"RNA": rna, "ATAC": atac}.items():
        if require_counts and "counts" not in adata_obj.layers:
            raise KeyError(f'{omics_name} AnnData is missing layers["counts"].')
        if require_cell_type and cell_type_key not in adata_obj.obs:
            raise KeyError(
                f'{omics_name} AnnData is missing obs["{cell_type_key}"].'
            )
        if not adata_obj.obs_names.is_unique:
            raise ValueError(f"{omics_name} observation names must be unique.")

    if require_paired and not rna.obs_names.equals(atac.obs_names):
        raise ValueError(
            "RNA and ATAC observation names must match in the same order for "
            f"{paired_task}."
        )

    summary_data = {
        "cells": [rna.n_obs, atac.n_obs],
        "features": [rna.n_vars, atac.n_vars],
    }
    if require_counts:
        summary_data["count_dtype"] = [
            rna.layers["counts"].dtype,
            atac.layers["counts"].dtype,
        ]
    if cell_type_key in rna.obs and cell_type_key in atac.obs:
        summary_data["cell_types"] = [
            rna.obs[cell_type_key].nunique(),
            atac.obs[cell_type_key].nunique(),
        ]
    input_summary = pd.DataFrame(summary_data, index=["RNA", "ATAC"])

    if close_backed:
        rna.file.close()
        atac.file.close()

    return rna, atac, graph_data, input_summary
