import torch
import torch.nn.functional as F

EPS = 1e-7


# ZINB 重构损失函数


# KL 散度损失


def info_nce_loss(z1, z2, temperature=0.3):
    # normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.shape[0]

    # similarity matrix
    sim = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(batch_size).to(z1.device)
    loss = F.cross_entropy(sim, labels)
    return loss


# def info_nce_cross(sim: torch.Tensor, pos_mask: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
#     """
#     """
#     logits = sim / tau
#     # large_neg = -1e9
#     neg_inf = torch.finfo(logits.dtype).min
#     pos_logits = logits.masked_fill(~pos_mask, neg_inf)
#     num = torch.logsumexp(pos_logits, dim=1)
#     den = torch.logsumexp(logits,     dim=1)
#     valid = pos_mask.any(dim=1)
#     return (-(num - den))[valid].mean() if valid.any() else logits.new_tensor(0.0)


def info_nce_cross(
    sim: torch.Tensor,
    pos_mask: torch.Tensor,
    tau: float = 0.1,
    symmetric: bool = True,
) -> torch.Tensor:
    """
    InfoNCE computed on the cross-domain similarity matrix.

    - Numerator: positives selected by pos_mask (supports multiple positives per row/col).
    - Denominator: all pairs in the corresponding row/col.
    - Optional symmetric form: compute both directions (rows as anchors and cols as anchors) and average.

    Args:
        sim:      [Ba, Bb] similarity matrix (e.g., cosine similarity).
        pos_mask: [Ba, Bb] boolean mask indicating positive pairs.
        tau:      temperature.
        symmetric: if True, compute both RNA->ATAC and ATAC->RNA and average.

    Returns:
        A scalar loss tensor.
    """
    logits = sim / tau
    # Use dtype-specific minimum to avoid fp16/bf16 overflow issues from large negative constants (e.g., -1e9).
    neg_inf = torch.finfo(logits.dtype).min

    def _one_way(_logits: torch.Tensor, _mask: torch.Tensor, dim: int) -> torch.Tensor:
        """
        One-direction NCE:
            loss = -log( sum_exp(pos) / sum_exp(all) )
        along the specified dimension.

        dim=1: row-wise (rows are anchors, columns are candidates)
        dim=0: col-wise (cols are anchors, rows are candidates)
        """
        # Mask out non-positives in the numerator.
        pos_logits = _logits.masked_fill(~_mask, neg_inf)

        # log(sum(exp(pos))) and log(sum(exp(all)))
        num = torch.logsumexp(pos_logits, dim=dim)
        den = torch.logsumexp(_logits, dim=dim)

        # Only keep anchors that have at least one positive.
        valid = _mask.any(dim=dim)
        return (-(num - den))[valid].mean() if valid.any() else _logits.new_tensor(0.0)

    # Direction 1: RNA -> ATAC (row-wise normalization)
    loss_row = _one_way(logits, pos_mask, dim=1)

    if not symmetric:
        return loss_row

    # Direction 2: ATAC -> RNA (col-wise normalization)
    loss_col = _one_way(logits, pos_mask, dim=0)

    # Symmetric average
    return 0.5 * (loss_row + loss_col)


"""
Joint_part – Loss Functions

L1  ReconstructionLoss   : masked NB (RNA) + masked BCE (ATAC)
L2  CrossModalInfoNCE    : paired RNA-view vs ATAC-view contrastive
L3  NeighborhoodConsist  : KL divergence to PCA-KNN soft targets
"""

# ── L1: Reconstruction ───────────────────────────────────────────────────────


# ── L2: Cross-modal InfoNCE ───────────────────────────────────────────────────


# ── L3: Neighbourhood consistency ────────────────────────────────────────────


################ part 3 joint newly
