# losses.py
import torch                
import torch.nn.functional as F
from typing import Optional, Dict, List

from src.evaluation.metrics import (
    DEFAULT_MAD_INV_CAP,
    proximity_mad_per_elem_torch,
    sparsity_l1_per_elem_torch,
    smoothness_per_elem_torch,
)


def proximity_loss(
    Delta: torch.Tensor,
    mad_inv: Optional[torch.Tensor] = None,
    feature_cost: Optional[torch.Tensor] = None,
    mad_inv_cap: float = DEFAULT_MAD_INV_CAP,
) -> torch.Tensor:
    """
    Feature-normalised Frobenius proximity loss.

    Delta: (N, T, D)
    mad_inv: (D,) optional inverse MAD scaling
    feature_cost: (D,) optional per-feature cost weight (higher => more expensive to change)
    """
    if feature_cost is not None:
        Delta = Delta * feature_cost.view(1, 1, -1)
    return proximity_mad_per_elem_torch(Delta, mad_inv=mad_inv, mad_inv_cap=mad_inv_cap)


def sparsity_loss(W: torch.Tensor) -> torch.Tensor:
    """
    L1 sparsity in basis coefficient space.
    W: (N, K, D)
    """
    return torch.mean(torch.sum(torch.abs(W), dim=(1, 2)))


def delta_sparsity_loss(Delta: torch.Tensor) -> torch.Tensor:
    """
    Canonical sparsity in input-space perturbation.
    Delta: (N, T, D)
    """
    return sparsity_l1_per_elem_torch(Delta)


def smoothness_loss(
    Delta: torch.Tensor,
    feature_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalise second-order finite differences over time.
    Delta: (N, T, D)
    feature_weight: (D,) optional weighting (e.g., stronger on state channels)
    """
    if feature_weight is not None:
        Delta = Delta * feature_weight.view(1, 1, -1)
    return smoothness_per_elem_torch(Delta)


def dpp_diversity_loss(W: torch.Tensor, jitter: float = 1e-2) -> torch.Tensor:
    """
    DPP diversity in W-space (stable log-det via Cholesky/SVD fallback).
    W: (N, K, D)
    """
    N = W.shape[0]
    if N <= 1:
        return torch.zeros((), device=W.device)

    w_flat = W.reshape(N, -1)
    w_norm = F.normalize(w_flat, p=2, dim=1, eps=1e-8)
    S = w_norm @ w_norm.t()  # cosine similarity kernel
    S = S + torch.eye(N, device=W.device) * jitter

    try:
        L = torch.linalg.cholesky(S)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        return -log_det
    except RuntimeError:
        s = torch.linalg.svdvals(S)
        return -torch.sum(torch.log(s.clamp_min(1e-8)))


def group_channel_sparsity_loss(Delta: torch.Tensor, groups: Dict[str, List[int]]) -> torch.Tensor:
    """
    Group-lasso style penalty on action channels/groups.
    groups example: {"insulin": [2,3], "diet": [4], "exercise": [5,6]}
    """
    if not groups:
        return torch.zeros((), device=Delta.device)

    vals = []
    for _, idxs in groups.items():
        if not idxs:
            continue
        g = Delta[:, :, idxs]  # (N, T, |g|)
        gn = torch.sqrt(torch.sum(g ** 2, dim=(1, 2)) + 1e-8)  # (N,)
        vals.append(gn)

    if not vals:
        return torch.zeros((), device=Delta.device)

    return torch.mean(torch.stack(vals, dim=1).sum(dim=1))


def validity_loss_regression(
    y_pred: torch.Tensor,
    target_value: Optional[float] = None,
    target_range: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Regression validity:
      - target_value: MSE to scalar target
      - target_range: hinge-squared to interval [lo, hi]
    """
    y = y_pred.squeeze(-1) if (y_pred.ndim > 1 and y_pred.shape[-1] == 1) else y_pred.squeeze()
    if y.ndim == 0:
        y = y.unsqueeze(0)

    if target_range is not None:
        lo, hi = target_range
        lo_t = torch.as_tensor(lo, dtype=y.dtype, device=y.device)
        hi_t = torch.as_tensor(hi, dtype=y.dtype, device=y.device)
        below = F.relu(lo_t - y)
        above = F.relu(y - hi_t)
        return torch.mean(below ** 2 + above ** 2)

    if target_value is None:
        raise ValueError("Provide target_value or target_range for regression validity.")

    y_t = torch.full_like(y, float(target_value))
    return torch.mean((y - y_t) ** 2)


def validity_loss_binary(
    logits_or_scores: torch.Tensor,
    target_class: int,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Binary validity.
    Supports:
      - shape (N,) or (N,1): single-logit BCEWithLogits
      - shape (N,2): CE
    """
    if logits_or_scores.ndim == 2 and logits_or_scores.shape[-1] == 2:
        y = torch.full((logits_or_scores.shape[0],), int(target_class),
                       device=logits_or_scores.device, dtype=torch.long)
        return F.cross_entropy(logits_or_scores, y)

    logits = logits_or_scores.squeeze(-1).squeeze()
    if logits.ndim == 0:
        logits = logits.unsqueeze(0)

    y = torch.full_like(logits, float(target_class))
    loss = F.binary_cross_entropy_with_logits(logits, y)

    if margin > 0:
        sign = 1.0 if int(target_class) == 1 else -1.0
        loss = loss + F.relu(margin - sign * logits).mean()

    return loss


def validity_loss_multiclass(
    logits: torch.Tensor,
    target_class: int,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    Multiclass validity with optional logit margin.
    logits: (N, C)
    """
    if logits.ndim != 2:
        raise ValueError("Expected logits of shape (N, C) for multiclass validity.")

    y = torch.full((logits.shape[0],), int(target_class), device=logits.device, dtype=torch.long)
    ce = F.cross_entropy(logits, y)

    if margin <= 0:
        return ce

    true_logits = logits[:, target_class]
    mask = F.one_hot(y, num_classes=logits.shape[1]).bool()
    max_other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
    margin_pen = F.relu(margin - (true_logits - max_other)).mean()
    return ce + margin_pen
