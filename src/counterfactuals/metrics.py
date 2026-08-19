import math
import numpy as np
import torch


def predict_batch(model, x_cf: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    x_cf: (N, T, D)
    returns: (N,) for scalar regression
    """
    model.eval()
    with torch.no_grad():
        y = model(x_cf.to(device))
        y = y.view(y.shape[0], -1).mean(dim=1)
    return y


def regression_band_violation_batch(pred: torch.Tensor, target) -> torch.Tensor:
    """
    pred: (N,)
    target: TargetSpec with target_range or target_value
    returns: (N,) nonnegative violation
    """
    if target.target_range is not None:
        lo, hi = target.target_range
    else:
        lo = hi = float(target.target_value)

    lo_t = pred.new_tensor(lo)
    hi_t = pred.new_tensor(hi)

    below = torch.relu(lo_t - pred)
    above = torch.relu(pred - hi_t)
    return below + above


def regression_point_gap_batch(pred: torch.Tensor, target) -> torch.Tensor:
    """
    Distance to centre target, useful for reporting.
    """
    if target.target_value is not None:
        centre = float(target.target_value)
    elif target.target_range is not None:
        centre = 0.5 * (float(target.target_range[0]) + float(target.target_range[1]))
    else:
        raise ValueError("TargetSpec must define target_value or target_range.")
    return torch.abs(pred - pred.new_tensor(centre))


# def proximity_from_delta(delta: torch.Tensor, mad_inv=None) -> torch.Tensor:
#     """
#     delta: (N, T, D)
#     returns scalar tensor
#     """
#     if mad_inv is not None:
#         mad_inv = torch.as_tensor(mad_inv, dtype=delta.dtype, device=delta.device).view(1, 1, -1)
#         delta = delta * mad_inv
#     return torch.mean(torch.sum(delta ** 2, dim=(1, 2)))

# def proximity_from_delta(delta: torch.Tensor, mad_inv=None, mode="raw_mean") -> torch.Tensor:
#     x = delta
#     if mad_inv is not None and mode == "mad_mean":
#         mad_inv = torch.as_tensor(mad_inv, dtype=delta.dtype, device=delta.device).view(1, 1, -1)
#         mad_inv = torch.clamp(mad_inv, max=50.0)
#         x = x * mad_inv
#     return torch.mean(x ** 2)

def proximity_from_delta(
    delta: torch.Tensor,
    mad_inv=None,
    mode: str = "raw_mean",
    mad_clip: float = 50.0,
) -> torch.Tensor:
    """
    delta: (N, T, D)

    mode:
      - "raw_mean":     mean(delta^2)
      - "mad_mean":     mean((delta * clipped_mad_inv)^2)
      - "raw_sum":      sum(delta^2) averaged over batch
      - "mad_sum":      sum((delta * clipped_mad_inv)^2) averaged over batch
    """
    x = delta

    if mad_inv is not None and mode in {"mad_mean", "mad_sum"}:
        mad_inv = torch.as_tensor(mad_inv, dtype=delta.dtype, device=delta.device).view(1, 1, -1)
        mad_inv = torch.clamp(mad_inv, max=mad_clip)
        x = x * mad_inv

    if mode in {"raw_mean", "mad_mean"}:
        return torch.mean(x ** 2)

    if mode in {"raw_sum", "mad_sum"}:
        return torch.mean(torch.sum(x ** 2, dim=(1, 2)))

    raise ValueError(f"Unknown proximity mode: {mode}")


def smoothness_from_delta(delta: torch.Tensor) -> torch.Tensor:
    """
    Mean squared second difference over time.
    delta: (N, T, D)
    """
    if delta.shape[1] < 3:
        return torch.zeros((), dtype=delta.dtype, device=delta.device)

    d2 = delta[:, 2:, :] - 2.0 * delta[:, 1:-1, :] + delta[:, :-2, :]
    return torch.mean(d2 ** 2)


def coeff_sparsity_from_W(W) -> torch.Tensor:
    """
    W: (N, K, D) or None
    """
    if W is None:
        return torch.tensor(float("nan"))
    if not isinstance(W, torch.Tensor):
        W = torch.as_tensor(W, dtype=torch.float32)
    return torch.mean(torch.sum(torch.abs(W), dim=(1, 2)))


def group_sparsity_from_delta(delta: torch.Tensor, action_groups: dict) -> torch.Tensor:
    """
    Group sparsity across channels/groups.
    delta: (N, T, D)
    action_groups: dict[str, list[int]]
    """
    if action_groups is None or len(action_groups) == 0:
        return torch.tensor(float("nan"), dtype=delta.dtype, device=delta.device)

    group_vals = []
    for _, idxs in action_groups.items():
        if len(idxs) == 0:
            continue
        g = delta[:, :, idxs]                           # (N, T, |g|)
        # group l2 norm per CF
        g_norm = torch.sqrt(torch.sum(g ** 2, dim=(1, 2)) + 1e-12)   # (N,)
        group_vals.append(g_norm)

    if len(group_vals) == 0:
        return torch.tensor(float("nan"), dtype=delta.dtype, device=delta.device)

    G = torch.stack(group_vals, dim=1)  # (N, num_groups)
    return torch.mean(torch.sum(G, dim=1))


def diversity_from_delta(delta: torch.Tensor, jitter: float = 1e-2) -> torch.Tensor:
    """
    Post hoc diversity computed from flattened perturbations.
    This is fair across all methods because all methods return X_cf.

    Returns a normalised score in [0, 1]:
        1.0 = maximally diverse (orthogonal perturbation directions)
        ~0  = all CFs identical
    Internally computes exp(logdet(K) / N) / (1 + jitter) where K is the
    unit-normalised Gram matrix + jitter * I.
    """
    N = delta.shape[0]
    if N <= 1:
        return torch.zeros((), dtype=delta.dtype, device=delta.device)

    flat = delta.reshape(N, -1)
    flat = flat / (torch.norm(flat, dim=1, keepdim=True) + 1e-8)
    K = flat @ flat.T
    K = K + jitter * torch.eye(N, device=delta.device, dtype=delta.dtype)

    sign, logabsdet = torch.linalg.slogdet(K)
    if sign <= 0:
        return torch.tensor(0.0, dtype=delta.dtype, device=delta.device)

    # Normalise: geometric mean of eigenvalues scaled by (1+jitter)
    # gives ~1 for orthogonal CFs, ~0 for identical CFs
    score = torch.exp(logabsdet / N) / (1.0 + jitter)
    return torch.clamp(score, min=0.0, max=1.0)


def evaluate_cf_set(
    model,
    x_orig,
    x_cf,
    target,
    schema,
    device,
    W=None,
    prox_mode: str = "raw_mean",
    prox_mad_clip: float = 50.0,
    clamp_to_schema: bool = True,
):
    if isinstance(x_orig, np.ndarray):
        x_orig = torch.tensor(x_orig, dtype=torch.float32, device=device)
    else:
        x_orig = x_orig.to(device=device, dtype=torch.float32)

    if isinstance(x_cf, np.ndarray):
        x_cf = torch.tensor(x_cf, dtype=torch.float32, device=device)
    else:
        x_cf = x_cf.to(device=device, dtype=torch.float32)

    if x_orig.ndim == 2:
        x_orig = x_orig.unsqueeze(0)
    if x_cf.ndim == 2:
        x_cf = x_cf.unsqueeze(0)

    # Optional post hoc clamp for fair evaluation across all methods
    if clamp_to_schema:
        if schema.min_vals is not None:
            mn = torch.as_tensor(schema.min_vals, dtype=x_cf.dtype, device=x_cf.device).view(1, 1, -1)
            x_cf = torch.maximum(x_cf, mn)
        if schema.max_vals is not None:
            mx = torch.as_tensor(schema.max_vals, dtype=x_cf.dtype, device=x_cf.device).view(1, 1, -1)
            x_cf = torch.minimum(x_cf, mx)

    N = x_cf.shape[0]
    x_orig_rep = x_orig.repeat(N, 1, 1)
    delta = x_cf - x_orig_rep

    pred = predict_batch(model, x_cf, device=device)
    validity_vec = regression_band_violation_batch(pred, target)
    point_gap_vec = regression_point_gap_batch(pred, target)

    validity = torch.mean(validity_vec)

    # Main paper metric: raw mean-square perturbation
    proximity_raw = proximity_from_delta(delta, mad_inv=None, mode="raw_mean")

    # Optional appendix/debug metric: MAD-weighted mean-square perturbation
    proximity_mad = proximity_from_delta(
        delta, mad_inv=schema.mad_inv, mode="mad_mean", mad_clip=prox_mad_clip
    )

    coeff_sparsity = coeff_sparsity_from_W(W)
    group_sparsity = group_sparsity_from_delta(delta, schema.action_groups)
    smoothness = smoothness_from_delta(delta)
    diversity = diversity_from_delta(delta)

    total = validity + proximity_raw
    if not torch.isnan(coeff_sparsity):
        total = total + coeff_sparsity
    if not torch.isnan(group_sparsity):
        total = total + group_sparsity
    total = total + smoothness

    is_valid = bool(torch.max(validity_vec).item() <= 1e-12)

    return {
        "validity": float(validity.item()),
        "point_gap": float(torch.mean(point_gap_vec).item()),
        "proximity": float(proximity_raw.item()),          # use this in tables
        "proximity_raw": float(proximity_raw.item()),
        "proximity_mad": float(proximity_mad.item()),
        "coeff_sparsity": float(coeff_sparsity.item()) if isinstance(coeff_sparsity, torch.Tensor) else float(coeff_sparsity),
        "group_sparsity": float(group_sparsity.item()) if isinstance(group_sparsity, torch.Tensor) else float(group_sparsity),
        "smoothness": float(smoothness.item()),
        "diversity": float(diversity.item()),
        "total": float(total.item()),
        "is_valid": float(is_valid),
        "pred_summary": float(torch.mean(pred).item()),
        "pred_all": pred.detach().cpu().numpy(),
    }

# def evaluate_cf_set(
#     model,
#     x_orig,               # (T, D) or (1, T, D)
#     x_cf,                 # (N, T, D) or (T, D)
#     target,
#     schema,
#     device,
#     W=None,               # optional, only basis method has this
# ):
#     """
#     Shared post hoc evaluator for ALL methods.
#     Returns a dict with the same keys you already use.
#     """
#     if isinstance(x_orig, np.ndarray):
#         x_orig = torch.tensor(x_orig, dtype=torch.float32, device=device)
#     else:
#         x_orig = x_orig.to(device=device, dtype=torch.float32)

#     if isinstance(x_cf, np.ndarray):
#         x_cf = torch.tensor(x_cf, dtype=torch.float32, device=device)
#     else:
#         x_cf = x_cf.to(device=device, dtype=torch.float32)

#     if x_orig.ndim == 2:
#         x_orig = x_orig.unsqueeze(0)   # (1, T, D)
#     if x_cf.ndim == 2:
#         x_cf = x_cf.unsqueeze(0)       # (N=1, T, D)

#     N = x_cf.shape[0]
#     x_orig_rep = x_orig.repeat(N, 1, 1)
#     delta = x_cf - x_orig_rep

#     pred = predict_batch(model, x_cf, device=device)                # (N,)
#     validity_vec = regression_band_violation_batch(pred, target)    # (N,)
#     point_gap_vec = regression_point_gap_batch(pred, target)        # (N,)

#     validity = torch.mean(validity_vec)
#     proximity = proximity_from_delta(delta, mad_inv=schema.mad_inv)
#     coeff_sparsity = coeff_sparsity_from_W(W)
#     group_sparsity = group_sparsity_from_delta(delta, schema.action_groups)
#     smoothness = smoothness_from_delta(delta)
#     diversity = diversity_from_delta(delta)

#     total = validity + proximity
#     if not torch.isnan(coeff_sparsity):
#         total = total + coeff_sparsity
#     if not torch.isnan(group_sparsity):
#         total = total + group_sparsity
#     total = total + smoothness

#     is_valid = bool(torch.max(validity_vec).item() <= 1e-12)

#     return {
#         "validity": float(validity.item()),                   # band violation
#         "point_gap": float(torch.mean(point_gap_vec).item()), # extra reporting metric
#         "proximity": float(proximity.item()),
#         "coeff_sparsity": float(coeff_sparsity.item()) if isinstance(coeff_sparsity, torch.Tensor) else float(coeff_sparsity),
#         "group_sparsity": float(group_sparsity.item()) if isinstance(group_sparsity, torch.Tensor) else float(group_sparsity),
#         "smoothness": float(smoothness.item()),
#         "diversity": float(diversity.item()),
#         "total": float(total.item()),
#         "is_valid": float(is_valid),
#         "pred_summary": float(torch.mean(pred).item()),
#         "pred_all": pred.detach().cpu().numpy(),
#     }