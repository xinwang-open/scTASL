# scTASL

**scTASL** is a task-structured deep learning framework for single-cell multi-omics analysis. Rather than relying on a single shared latent space, scTASL coordinates purpose-built representations across five analytical tasks — integration, imputation, cross-omics translation, clustering, and gene regulatory network (GRN) inference — enabling a more complete picture of cellular heterogeneity and regulatory programs from paired scRNA-seq and scATAC-seq data.

## Features

| Task | Notebook | Description |
|------|----------|-------------|
| Integration | `1_integration.ipynb` | Joint cell and feature embedding for paired RNA + ATAC |
| Imputation & Translation | `2_imputation_translation.ipynb` | Intra-omics denoising and cross-omics profile generation |
| Multi-omics Clustering | `3_multi-omics_clustering.ipynb` | Graph-based clustering with Leiden algorithm |
| Fine-tuning | `4.1_integration_fine-tuning.ipynb` | Cell-type-specific embedding refinement |
| GRN Inference | `4.2_GRN_inference.ipynb` | TF–target gene network inference with cis-regulatory evidence |

## Installation

We recommend using a dedicated conda environment.

```bash
conda create -n sctasl python=3.10
conda activate sctasl
```

Install [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) matching your CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `pybedtools` requires a system installation of [bedtools](https://bedtools.readthedocs.io/en/latest/). Install it via `conda install -c bioconda bedtools` or your system package manager before running GRN inference.

## Data Preparation

This repository does not ship any dataset. Each tutorial expects preprocessed data under `dataset/<DATASET>/`, which you should download and prepare yourself:

| File | Description |
|------|-------------|
| `rna_processed.h5ad` | Processed RNA count matrix with a `counts` layer |
| `atac_processed.h5ad` | Processed ATAC count matrix with a `counts` layer |
| `graph_data.pkl` | Precomputed gene–peak prior graph |

For `PBMC-10k`, the raw 10x Multiome dataset is available from the official [10x Genomics dataset page](https://support.10xgenomics.com/single-cell-multiome-atac-gex/datasets/1.0.0/pbmc_granulocyte_sorted_10k). Pre-converted `.h5ad` files, along with the JASPAR motif tracks used for GRN inference below, are mirrored by [GLUE](https://scglue.readthedocs.io/en/latest/data.html) — a related single-cell multi-omics integration tool whose data-preprocessing conventions this repo follows:
- RNA: http://ftp.cbi.pku.edu.cn/pub/GLUE/dataset/10x-Multiome-Pbmc10k-RNA.h5ad
- ATAC: http://ftp.cbi.pku.edu.cn/pub/GLUE/dataset/10x-Multiome-Pbmc10k-ATAC.h5ad

See `dataset/PBMC-10k/data_preprocessing.ipynb` for how these are turned into `rna_processed.h5ad`, `atac_processed.h5ad`, and `graph_data.pkl`.

GENCODE annotation ([human v43](https://www.gencodegenes.org/human/release_43.html), [mouse vM25](https://www.gencodegenes.org/mouse/release_M25.html)) is also needed — during preprocessing, to supply gene coordinates for datasets that don't already provide them, and during GRN inference (`4.2_GRN_inference.ipynb`):
- https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gencode.v43.annotation.gtf.gz
- https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz


> Cell-type labels (`obs["cell_type"]`) are used only for evaluation and visualization. They are **not** required to train any model.

## Tutorials

All notebooks are designed to run from the **repository root** so that relative imports resolve correctly:

```bash
git clone https://github.com/xinwang-open/scTASL.git
cd scTASL
jupyter lab
```

Run the notebooks in order for the full workflow:

1. `1_integration.ipynb` — train the integration model and generate joint embeddings
2. `2_imputation_translation.ipynb` — impute and translate between modalities
3. `3_multi-omics_clustering.ipynb` — perform multi-omics clustering
4. `4.1_integration_fine-tuning.ipynb` — fine-tune embeddings for a specific cell type
5. `4.2_GRN_inference.ipynb` — infer a cell-type-specific TF–target gene network

## Citation

The manuscript describing scTASL has not yet been published. Citation information will be added here once it is available.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
