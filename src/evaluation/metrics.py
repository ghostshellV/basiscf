# ─────────────────────────────────────────────────────────────────────────────
# Unified Counterfactual Evaluation Metrics
#
# This module provides a SINGLE evaluation function used by ALL generators
# (BasisGenerator, CoMTEGenerator, ForecastCFGenerator) for fair comparison.
#
# Key design decisions:
#   - All metrics use per-element normalisation (mean, not sum) so values
#     are independent of sequence length T and feature dimension D.
#   - mad_inv is capped to prevent extreme scaling from near-zero MAD features.
#   - Every generator's CFs are evaluated post-hoc through this same function.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
from typing import Dict, Optional


# ── Default cap for mad_inv ──────────────────────────────────────────────────
# Features with near-zero MAD produce mad_inv ~ 1e8, which makes proximity
# explode for any method that touches them. Capping at a reasonable value
# (e.g. 100) prevents a single low-variance feature from dominating the metric.
DEFAULT_MAD_INV_CAP = 100.0
DEFAULT_FEATURE_THRESHOLD = 0.001
DEFAULT_IMMUTABLE_TOL = 1e-6

# Canonical contract metadata shared by Basis/CoMTE/Forecast generators.
CANONICAL_METRICS_SCHEMA_VERSION = "v1"
CANONICAL_HISTORY_REQUIRED_KEYS = (
    "iter",
    "total_loss",
    "l_valid",
    "l_prox",
    "l_sparse",
    "l_smooth",
    "validity_err",
)
CANONICAL_INFO_REQUIRED_KEYS = (
    "metrics_schema_version",
    "history_key_contract",
    "best_validity_err",
    "best_total_loss",
    "history",
    "history_by_run",
    "cf_pred",
    "error",
    "success",
    "all_cf_preds",
    "weights",
    "feature_names",
    "roles",
)


def get_canonical_history_key_contract() -> Dict[str, object]:
    """Return canonical key contract for generator history payloads."""
    return {
        "required": list(CANONICAL_HISTORY_REQUIRED_KEYS),
        "optional": [
            "l_group",
            "l_div",
            "l_temporal",
            "l_weighted_mae",
            "v_err",  # backward-compatible alias for validity_err
        ],
    }


def get_canonical_info_key_contract() -> Dict[str, object]:
    """Return canonical key contract for generator info payloads."""
    return {
        "required": list(CANONICAL_INFO_REQUIRED_KEYS),
        "notes": {
            "history": "Canonical history for the primary run (run 0).",
            "history_by_run": "List of per-run history lists; length equals num_cfs runs.",
            "error": "Primary-run validity error in normalized target space.",
            "weights": "Basis coefficients when available (BasisGenerator), else None.",
        },
    }


def _clip_mad_inv_np(
    mad_inv: Optional[np.ndarray],
    mad_inv_cap: float = DEFAULT_MAD_INV_CAP,
) -> Optional[np.ndarray]:
    if mad_inv is None:
        return None
    return np.clip(np.asarray(mad_inv, dtype=np.float32), 0.0, mad_inv_cap)


def _clip_mad_inv_torch(
    mad_inv: Optional[torch.Tensor],
    mad_inv_cap: float = DEFAULT_MAD_INV_CAP,
) -> Optional[torch.Tensor]:
    if mad_inv is None:
        return None
    return torch.clamp(mad_inv, min=0.0, max=mad_inv_cap)


def validity_regression_mse(pred: float, target_value: float) -> float:
    """Canonical regression validity MSE in scalar space."""
    return float((pred - target_value) ** 2)


def validity_regression_mae(pred: float, target_value: float) -> float:
    """Canonical regression validity MAE in scalar space."""
    return float(abs(pred - target_value))


def proximity_l2_per_elem_np(delta: np.ndarray) -> float:
    """Canonical unscaled proximity: mean(delta^2)."""
    return float(np.mean(delta ** 2))


def proximity_mad_per_elem_np(
    delta: np.ndarray,
    mad_inv: Optional[np.ndarray],
    mad_inv_cap: float = DEFAULT_MAD_INV_CAP,
) -> float:
    """Canonical MAD-scaled proximity: mean((delta * mad_inv)^2)."""
    if mad_inv is None:
        return proximity_l2_per_elem_np(delta)
    mad_inv_safe = _clip_mad_inv_np(mad_inv, mad_inv_cap=mad_inv_cap)
    delta_scaled = delta * mad_inv_safe[np.newaxis, :]
    return float(np.mean(delta_scaled ** 2))


def sparsity_l1_per_elem_np(delta: np.ndarray) -> float:
    """Canonical sparsity in input space: mean(abs(delta))."""
    return float(np.mean(np.abs(delta)))


def feature_sparsity_np(
    delta: np.ndarray,
    feature_threshold: float = DEFAULT_FEATURE_THRESHOLD,
) -> float:
    """Fraction of features with mean absolute change below threshold."""
    feat_change = np.mean(np.abs(delta), axis=0)
    return float(np.mean(feat_change < feature_threshold))


def temporal_sparsity_np(
    delta: np.ndarray,
    feature_threshold: float = DEFAULT_FEATURE_THRESHOLD,
) -> float:
    """Fraction of timesteps with mean absolute change below threshold."""
    abs_delta_per_t = np.mean(np.abs(delta), axis=1)
    return float(np.mean(abs_delta_per_t < feature_threshold))


def smoothness_per_elem_np(delta: np.ndarray) -> float:
    """Canonical smoothness: mean squared second-order finite difference."""
    if delta.shape[0] < 3:
        return 0.0
    d2 = delta[2:, :] - 2 * delta[1:-1, :] + delta[:-2, :]
    return float(np.mean(d2 ** 2))


def actionability_np(
    delta: np.ndarray,
    mutable_mask: Optional[np.ndarray],
    immutable_tol: float = DEFAULT_IMMUTABLE_TOL,
) -> float:
    """Returns 1.0 if immutable features are unchanged, else 0.0."""
    if mutable_mask is None:
        return 1.0
    mutable_mask = np.asarray(mutable_mask, dtype=bool)
    immutable_mask = ~mutable_mask
    # If there are no immutable channels, actionability is trivially satisfied.
    if not np.any(immutable_mask):
        return 1.0
    immutable_delta = float(np.max(np.abs(delta[:, immutable_mask])))
    return 1.0 if immutable_delta < immutable_tol else 0.0


def in_bounds_np(
    cf: np.ndarray,
    lower: float = -0.01,
    upper: float = 1.01,
) -> float:
    """Returns 1.0 if CF values stay inside [lower, upper], else 0.0."""
    return 1.0 if (np.all(cf >= lower) and np.all(cf <= upper)) else 0.0


def proximity_l2_per_elem_torch(delta: torch.Tensor) -> torch.Tensor:
    """Torch variant of canonical unscaled proximity: mean(delta^2)."""
    return torch.mean(delta ** 2)


def proximity_mad_per_elem_torch(
    delta: torch.Tensor,
    mad_inv: Optional[torch.Tensor],
    mad_inv_cap: float = DEFAULT_MAD_INV_CAP,
) -> torch.Tensor:
    """Torch variant of canonical MAD-scaled proximity: mean((delta * mad_inv)^2)."""
    if mad_inv is None:
        return proximity_l2_per_elem_torch(delta)
    mad_inv_safe = _clip_mad_inv_torch(mad_inv, mad_inv_cap=mad_inv_cap)
    scaled = delta * mad_inv_safe.view(1, 1, -1)
    return torch.mean(scaled ** 2)


def sparsity_l1_per_elem_torch(delta: torch.Tensor) -> torch.Tensor:
    """Torch variant of canonical sparsity in input space: mean(abs(delta))."""
    return torch.mean(torch.abs(delta))


def smoothness_per_elem_torch(delta: torch.Tensor) -> torch.Tensor:
    """Torch variant of canonical smoothness: mean squared second-order finite diff."""
    if delta.shape[1] < 3:
        return torch.zeros((), dtype=delta.dtype, device=delta.device)
    d2 = delta[:, 2:, :] - 2 * delta[:, 1:-1, :] + delta[:, :-2, :]
    return torch.mean(d2 ** 2)


def compute_evaluation_metrics(
    model: torch.nn.Module,
    original: np.ndarray,       # (T, D)
    cf: np.ndarray,             # (T, D)
    target_value: float,        # normalised target
    mad_inv: Optional[np.ndarray] = None,   # (D,)
    mad_inv_cap: float = DEFAULT_MAD_INV_CAP,
    mutable_mask: Optional[np.ndarray] = None,  # (D,) bool — True = editable
    device: Optional[torch.device] = None,
    feature_threshold: float = DEFAULT_FEATURE_THRESHOLD,
) -> Dict[str, float]:
    """
    Compute standardised evaluation metrics for a single counterfactual.

    All proximity / sparsity / smoothness metrics use **per-element mean**
    (not sum over T*D) so values are independent of sequence shape.

    Parameters
    ----------
    model : torch.nn.Module
        The predictive model (frozen, eval mode).
    original : np.ndarray, shape (T, D)
        Original input sequence.
    cf : np.ndarray, shape (T, D)
        Counterfactual sequence.
    target_value : float
        Desired model output (normalised).
    mad_inv : np.ndarray, shape (D,), optional
        Inverse MAD per feature. Will be capped at ``mad_inv_cap``.
    mad_inv_cap : float
        Maximum allowed mad_inv value (default 100).
    mutable_mask : np.ndarray, shape (D,), optional
        Boolean mask indicating which features are editable.
    device : torch.device, optional
        Device for model inference.
    feature_threshold : float
        Threshold below which a feature is considered unchanged (for sparsity).

    Returns
    -------
    dict with keys:
        validity_mse, validity_mae,
        proximity_l2_per_elem, proximity_mad_per_elem,
        sparsity_l1_per_elem, feature_sparsity, temporal_sparsity,
        smoothness_per_elem,
        actionability (1.0 if immutable features unchanged, else 0.0),
        in_bounds (1.0 if CF in [0,1] bounds, else 0.0),
        cf_pred
    """
    if device is None:
        device = next(model.parameters()).device

    T, D = original.shape
    delta = cf - original   # (T, D)

    # ── 1. Validity ──────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(cf, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x_t).cpu().item()

    validity_mse = validity_regression_mse(pred, target_value)
    validity_mae = validity_regression_mae(pred, target_value)

    # ── 2. Proximity (per-element) ───────────────────────────────────────
    proximity_l2 = proximity_l2_per_elem_np(delta)
    proximity_mad = proximity_mad_per_elem_np(delta, mad_inv=mad_inv, mad_inv_cap=mad_inv_cap)

    # ── 3. Sparsity (per-element L1 + count-based) ───────────────────────
    sparsity_l1 = sparsity_l1_per_elem_np(delta)
    feature_sparsity = feature_sparsity_np(delta, feature_threshold=feature_threshold)
    temporal_sparsity = temporal_sparsity_np(delta, feature_threshold=feature_threshold)

    # ── 4. Smoothness (per-element, 2nd-order finite diff) ───────────────
    smoothness = smoothness_per_elem_np(delta)

    # ── 5. Actionability ─────────────────────────────────────────────────
    actionable = actionability_np(delta, mutable_mask=mutable_mask)

    # ── 6. Plausibility (within [0, 1]) ──────────────────────────────────
    in_bounds = in_bounds_np(cf)

    return {
        'validity_mse': validity_mse,
        'validity_mae': validity_mae,
        'proximity_l2_per_elem': proximity_l2,
        'proximity_mad_per_elem': proximity_mad,
        'sparsity_l1_per_elem': sparsity_l1,
        'feature_sparsity': feature_sparsity,
        'temporal_sparsity': temporal_sparsity,
        'smoothness_per_elem': smoothness,
        'actionable': actionable,
        'in_bounds': in_bounds,
        'cf_pred': pred,
    }


def evaluate_generator(
    model: torch.nn.Module,
    gen_cfs: Dict[int, dict],
    mad_inv: Optional[np.ndarray] = None,
    mad_inv_cap: float = DEFAULT_MAD_INV_CAP,
    mutable_mask: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    rul_min: float = 0.0,
    rul_range: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate all CFs from one generator, returning aggregated metrics.

    Parameters
    ----------
    gen_cfs : dict
        {sample_idx: {'original': (T,D), 'cf': (T,D), 'target_rul': float, ...}}
    """
    all_metrics = []

    for idx, d in gen_cfs.items():
        target_norm = (d['target_rul'] - rul_min) / rul_range
        m = compute_evaluation_metrics(
            model=model,
            original=d['original'],
            cf=d['cf'],
            target_value=target_norm,
            mad_inv=mad_inv,
            mad_inv_cap=mad_inv_cap,
            mutable_mask=mutable_mask,
            device=device,
        )
        all_metrics.append(m)

    if not all_metrics:
        return {}

    # Aggregate: mean over all samples
    keys = all_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        result[f'{k}_mean'] = float(np.mean(vals))
        result[f'{k}_std'] = float(np.std(vals))

    return result
