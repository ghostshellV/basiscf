from __future__ import annotations

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from .basis import BSplineBasis, FourierBasis, RBFBasis, WaveletBasis, PolynomialBasis
from .losses import (
    proximity_loss,
    sparsity_loss,
    smoothness_loss,
    dpp_diversity_loss,
    group_channel_sparsity_loss,
)

FeatureRole = Literal["immutable", "action", "state", "context"]
TaskType = Literal["regression", "binary", "multiclass"]


# ---------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------

@dataclass
class TSFeatureSchema:
    feature_names: List[str]
    roles: List[FeatureRole]

    mutable_mask: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None
    min_vals: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None
    max_vals: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None

    mad_inv: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None
    change_cost: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None

    time_mutable_mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    static_mask: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None

    step_size: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None
    integer_mask: Optional[Union[np.ndarray, torch.Tensor, List[float]]] = None

    action_groups: Dict[str, List[int]] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.feature_names) != len(self.roles):
            raise ValueError("feature_names and roles must have the same length.")

    @property
    def D(self) -> int:
        return len(self.feature_names)


@dataclass
class TargetSpec:
    task_type: TaskType

    # Regression
    target_value: Optional[float] = None
    target_range: Optional[Tuple[float, float]] = None

    # Classification
    target_class: Optional[int] = None
    margin: float = 0.0


@dataclass
class GeneratorConfig:
    basis_type: str = "bspline"
    num_basis: int = 8
    device: str = "cpu"

    editable_roles: Tuple[FeatureRole, ...] = ("action",)
    allow_state_edits: bool = False

    init_std: float = 2e-2
    lr: float = 2e-2
    max_iter: int = 600
    eta_min: float = 1e-4
    gradient_clip_norm: float = 1.0
    early_stop_tol_reg: float = 1e-2
    early_stop_patience: int = 120
    clamp_during_optim: bool = True

    # New optimisation controls
    num_restarts: int = 5
    adam_steps: int = 300
    lbfgs_steps: int = 40

    # Augmented validity penalty
    rho_init: float = 10.0
    rho_growth: float = 1.4
    rho_max: float = 1e4

    # Before first valid CF, down-weight cleanup penalties
    prevalid_cleanup_scale: float = 0.15

    # Start diversity only after part of the run
    div_start_frac: float = 0.50


@dataclass
class LossWeights:
    validity: float = 1.0
    proximity: float = 2.0
    sparsity: float = 0.05          # coefficient sparsity on W
    diversity: float = 0.02
    smoothness: float = 0.50
    channel_sparsity: float = 0.75  # group/channel sparsity on Delta
    state_lock: float = 0.0


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _to_tensor(x, device, dtype=torch.float32):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class DefaultOutputAdapter:
    """
    Adapts arbitrary model output to a tensor.
    Supports:
      - tensor
      - tuple/list -> first element
      - dict with 'logits' or 'pred'
    """
    def __call__(self, y_raw: Any) -> torch.Tensor:
        if isinstance(y_raw, torch.Tensor):
            return y_raw
        if isinstance(y_raw, (tuple, list)):
            if len(y_raw) == 0:
                raise ValueError("Model returned an empty tuple/list.")
            if not isinstance(y_raw[0], torch.Tensor):
                raise ValueError("First output element is not a tensor.")
            return y_raw[0]
        if isinstance(y_raw, dict):
            if "logits" in y_raw and isinstance(y_raw["logits"], torch.Tensor):
                return y_raw["logits"]
            if "pred" in y_raw and isinstance(y_raw["pred"], torch.Tensor):
                return y_raw["pred"]
            raise ValueError("Unsupported dict output. Expected keys: 'logits' or 'pred'.")
        raise ValueError("Unsupported model output type.")


# ---------------------------------------------------------------------
# Basis CF generator
# ---------------------------------------------------------------------

class BasisGenerator:
    """
    Basis-guided counterfactual generator for multivariate time series.

    Main design:
      1. Optimise coefficients W in a low-dimensional basis space.
      2. Treat validity as a target-band / constraint-style penalty.
      3. Use channel/group sparsity rather than pointwise delta sparsity.
      4. Run Adam first, then LBFGS refinement.
      5. Select the best CF lexicographically:
         valid first, then proximity, then channel sparsity, then smoothness.
    """

    def __init__(
        self,
        model: nn.Module,
        sequence_length: int,
        feature_dim: int,
        basis_type: str = "bspline",
        num_basis: int = 8,
        device: str = "cpu",
        output_adapter: Optional[Callable[[Any], torch.Tensor]] = None,
        sequence_key: Optional[str] = None,
        config: Optional[GeneratorConfig] = None,
    ):
        self.model = model
        self.sequence_length = int(sequence_length)
        self.feature_dim = int(feature_dim)
        self.sequence_key = sequence_key

        self.config = copy.deepcopy(config) if config is not None else GeneratorConfig()
        self.device = torch.device(device if device is not None else self.config.device)

        self.basis_type = basis_type or self.config.basis_type
        self.num_basis = int(num_basis if num_basis is not None else self.config.num_basis)

        self.output_adapter = output_adapter or DefaultOutputAdapter()

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.Phi = self._build_basis(self.basis_type, self.sequence_length, self.num_basis).detach()
        self.last_history_: List[Dict[str, float]] = []

    # ------------------------- basis -------------------------

    def _build_basis(self, basis_type: str, T: int, K: int) -> torch.Tensor:
        basis_type = basis_type.lower()
        if basis_type == "bspline":
            basis = BSplineBasis(T, K)
        elif basis_type == "fourier":
            basis = FourierBasis(T, K)
        elif basis_type == "rbf":
            basis = RBFBasis(T, K)
        elif basis_type == "wavelet":
            basis = WaveletBasis(T, K)
        elif basis_type == "polynomial":
            basis = PolynomialBasis(T, K)
        else:
            raise ValueError(f"Unknown basis_type='{basis_type}'")
        basis = basis.to(self.device)
        with torch.no_grad():
            Phi = basis().to(self.device)
        return Phi

    # ------------------------- input helpers -------------------------

    def _extract_sequence(self, query_instance: Union[torch.Tensor, np.ndarray, Dict[str, Any]]) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if isinstance(query_instance, dict):
            if self.sequence_key is None:
                raise ValueError("sequence_key must be set when query_instance is a dict.")
            x = query_instance[self.sequence_key]
            x = _to_tensor(x, self.device)
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim != 2:
                raise ValueError(f"Expected perturbable sequence of shape (T, D), got {tuple(x.shape)}.")
            return x, query_instance
        x = _to_tensor(query_instance, self.device)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2:
            raise ValueError(f"Expected query_instance of shape (T, D), got {tuple(x.shape)}.")
        return x, None

    def _repeat_aux_tensor(self, t: torch.Tensor, batch_size: int) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == batch_size:
            return t
        if t.shape[0] == 1:
            return t.repeat(batch_size, *([1] * (t.ndim - 1)))
        return t.unsqueeze(0).repeat(batch_size, *([1] * t.ndim))

    def _rebuild_model_input(
        self,
        original_payload: Optional[Dict[str, Any]],
        x_cf: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if original_payload is None:
            return x_cf

        out: Dict[str, torch.Tensor] = {}
        for k, v in original_payload.items():
            if k == self.sequence_key:
                out[k] = x_cf
            else:
                t = _to_tensor(v, self.device)
                out[k] = self._repeat_aux_tensor(t, x_cf.shape[0])
        return out

    # ------------------------- masks and bounds -------------------------

    def _feature_mutable_mask(self, schema: TSFeatureSchema) -> torch.Tensor:
        D = schema.D
        if schema.mutable_mask is not None:
            m = _to_tensor(schema.mutable_mask, self.device).view(-1) > 0.5
            if m.numel() != D:
                raise ValueError("schema.mutable_mask must have shape (D,)")
            return m

        vals = []
        for role in schema.roles:
            editable = role in self.config.editable_roles
            if role == "state" and self.config.allow_state_edits:
                editable = True
            if role in ("immutable", "context"):
                editable = False
            vals.append(editable)
        return torch.tensor(vals, device=self.device, dtype=torch.bool)

    def _time_feature_mutable_mask(self, schema: TSFeatureSchema, T: int, D: int) -> torch.Tensor:
        fmask = self._feature_mutable_mask(schema).view(1, D).expand(T, D)

        if schema.time_mutable_mask is None:
            return fmask

        tmask = _to_tensor(schema.time_mutable_mask, self.device)
        if tmask.ndim == 1:
            tmask = tmask.view(T, 1).expand(T, D)
        elif tmask.ndim == 2:
            if tmask.shape == (T, 1):
                tmask = tmask.expand(T, D)
            elif tmask.shape == (1, D):
                tmask = tmask.expand(T, D)
            elif tmask.shape != (T, D):
                raise ValueError("time_mutable_mask must have shape (T,), (T,1), (1,D), or (T,D)")
        else:
            raise ValueError("time_mutable_mask must be 1D or 2D")
        return fmask & (tmask > 0.5)

    def _bounds(self, schema: TSFeatureSchema, T: int, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = torch.full((1, T, D), -3.0, device=self.device)
        upper = torch.full((1, T, D),  3.0, device=self.device)

        if schema.min_vals is not None:
            mn = _to_tensor(schema.min_vals, self.device)
            if mn.ndim == 1:
                mn = mn.view(1, 1, D).expand(1, T, D)
            elif mn.ndim == 2:
                mn = mn.view(1, T, D)
            lower = mn

        if schema.max_vals is not None:
            mx = _to_tensor(schema.max_vals, self.device)
            if mx.ndim == 1:
                mx = mx.view(1, 1, D).expand(1, T, D)
            elif mx.ndim == 2:
                mx = mx.view(1, T, D)
            upper = mx

        return lower, upper

    def _apply_constraints(
        self,
        X_orig: torch.Tensor,
        Delta: torch.Tensor,
        schema: TSFeatureSchema,
        final_pass: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, T, D = Delta.shape

        mutable = self._time_feature_mutable_mask(schema, T, D).view(1, T, D)
        Delta = Delta * mutable

        # Static repeated features: same edit across time
        if schema.static_mask is not None:
            sm = _to_tensor(schema.static_mask, self.device).view(1, 1, D) > 0.5
            if sm.any():
                mean_delta = Delta.mean(dim=1, keepdim=True)
                Delta = torch.where(sm, mean_delta.expand_as(Delta), Delta)

        X_cf = X_orig + Delta

        lower, upper = self._bounds(schema, T, D)
        if self.config.clamp_during_optim or final_pass:
            X_cf = torch.max(torch.min(X_cf, upper), lower)

        # Optional discretisation only at final pass
        if final_pass:
            if schema.step_size is not None:
                step = _to_tensor(schema.step_size, self.device).view(1, 1, D)
                use = step > 0
                X_cf = torch.where(use, torch.round(X_cf / step.clamp_min(1e-8)) * step, X_cf)

            if schema.integer_mask is not None:
                im = _to_tensor(schema.integer_mask, self.device).view(1, 1, D) > 0.5
                X_cf = torch.where(im, torch.round(X_cf), X_cf)

            X_cf = torch.max(torch.min(X_cf, upper), lower)

        Delta = X_cf - X_orig
        return X_cf, Delta

    # ------------------------- prediction + validity -------------------------

    def _predict_tensor(self, model_input: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        y_raw = self.model(model_input)
        y = self.output_adapter(y_raw)
        if not isinstance(y, torch.Tensor):
            raise ValueError("output_adapter must return a tensor.")
        return y

    def _regression_scalar(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 0:
            y = y.view(1)
        y = y.reshape(y.shape[0], -1)
        return y.mean(dim=1)

    def _regression_violation(self, y: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        pred = self._regression_scalar(y)
        if target.target_range is not None:
            lo, hi = target.target_range
        elif target.target_value is not None:
            tol = float(self.config.early_stop_tol_reg)
            lo = float(target.target_value) - tol
            hi = float(target.target_value) + tol
        else:
            raise ValueError("Regression target must provide target_value or target_range.")
        lo_t = pred.new_tensor(lo)
        hi_t = pred.new_tensor(hi)
        return torch.relu(lo_t - pred) + torch.relu(pred - hi_t)

    def _binary_violation(self, y: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        margin = float(target.margin)
        if target.target_class is None:
            raise ValueError("Binary target_class must be provided.")

        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            logit = y.view(-1)
            if int(target.target_class) == 1:
                return torch.relu(margin - logit)
            return torch.relu(margin + logit)

        logits = y.reshape(y.shape[0], -1)
        if logits.shape[1] != 2:
            raise ValueError("Binary task expected 1 logit or 2 logits.")
        t = int(target.target_class)
        other = 1 - t
        return torch.relu(logits[:, other] - logits[:, t] + margin)

    def _multiclass_violation(self, y: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        if target.target_class is None:
            raise ValueError("Multiclass target_class must be provided.")
        logits = y.reshape(y.shape[0], -1)
        t = int(target.target_class)
        margin = float(target.margin)
        target_logit = logits[:, t]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, t] = False
        max_other = logits.masked_fill(~mask, -1e9).max(dim=1).values
        return torch.relu(max_other - target_logit + margin)

    def _validity_violation(self, y: torch.Tensor, target: TargetSpec) -> torch.Tensor:
        if target.task_type == "regression":
            return self._regression_violation(y, target)
        if target.task_type == "binary":
            return self._binary_violation(y, target)
        if target.task_type == "multiclass":
            return self._multiclass_violation(y, target)
        raise ValueError(f"Unknown task_type={target.task_type}")

    def _prediction_summary(self, y: torch.Tensor, target: TargetSpec) -> float:
        if target.task_type == "regression":
            return float(self._regression_scalar(y).mean().item())
        if y.ndim == 1:
            return float(y.mean().item())
        logits = y.reshape(y.shape[0], -1)
        return float(logits.argmax(dim=1).float().mean().item())

    # ------------------------- objective -------------------------

    def _compute_metrics(
        self,
        X_orig: torch.Tensor,
        W: torch.Tensor,
        schema: TSFeatureSchema,
        target: TargetSpec,
        num_cfs: int,
        loss_weights: LossWeights,
        rho: float,
        cleanup_scale: float,
        step_frac: float,
    ) -> Dict[str, Any]:
        Delta = torch.einsum("tk,nkd->ntd", self.Phi, W)
        X_cf, Delta = self._apply_constraints(X_orig, Delta, schema, final_pass=False)

        model_input = self._rebuild_model_input(None, X_cf)
        y = self._predict_tensor(model_input)
        violation = self._validity_violation(y, target)
        L_valid = violation.mean()

        mad_inv = _to_tensor(schema.mad_inv, self.device)
        change_cost = _to_tensor(schema.change_cost, self.device)

        L_prox = proximity_loss(Delta, mad_inv=mad_inv, feature_cost=change_cost)
        L_coef_sparse = sparsity_loss(W)
        L_smooth = smoothness_loss(Delta)

        if loss_weights.channel_sparsity > 0.0 and schema.action_groups:
            L_group = group_channel_sparsity_loss(Delta, schema.action_groups)
        else:
            L_group = torch.zeros((), device=self.device)

        state_idx = [i for i, r in enumerate(schema.roles) if r == "state"]
        if loss_weights.state_lock > 0.0 and len(state_idx) > 0:
            L_state = torch.mean(torch.abs(Delta[:, :, state_idx]))
        else:
            L_state = torch.zeros((), device=self.device)

        if num_cfs > 1 and loss_weights.diversity > 0.0 and step_frac >= self.config.div_start_frac:
            L_div = dpp_diversity_loss(W)
        else:
            L_div = torch.zeros((), device=self.device)

        cleanup = (
            loss_weights.proximity * L_prox
            + loss_weights.sparsity * L_coef_sparse
            + loss_weights.smoothness * L_smooth
            + loss_weights.channel_sparsity * L_group
            + loss_weights.state_lock * L_state
            + loss_weights.diversity * L_div
        )

        total = loss_weights.validity * L_valid + rho * (L_valid ** 2) + cleanup_scale * cleanup
        is_valid = bool(torch.max(violation).item() <= 1e-8)

        return {
            "total": total,
            "X_cf": X_cf,
            "Delta": Delta,
            "pred_tensor": y,
            "pred_summary": self._prediction_summary(y, target),
            "is_valid": is_valid,
            "validity": float(L_valid.detach().item()),
            "proximity": float(L_prox.detach().item()),
            "coeff_sparsity": float(L_coef_sparse.detach().item()),
            "group_sparsity": float(L_group.detach().item()),
            "smoothness": float(L_smooth.detach().item()),
            "diversity": float(L_div.detach().item()),
            "state_lock": float(L_state.detach().item()),
        }

    def _selection_key(self, m: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        # valid first, then validity error, then proximity, then group sparsity, then smoothness
        invalid = 0.0 if m["is_valid"] else 1.0
        return (
            invalid,
            m["validity"],
            m["proximity"],
            m["group_sparsity"],
            m["smoothness"],
        )

    # ------------------------- public API -------------------------

    def generate(
        self,
        query_instance: Union[torch.Tensor, np.ndarray, Dict[str, Any]],
        target: TargetSpec,
        schema: TSFeatureSchema,
        num_cfs: int = 1,
        loss_weights: Optional[LossWeights] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if target is None:
            raise ValueError("target must be provided.")
        if schema.D != self.feature_dim:
            raise ValueError(f"schema.D={schema.D} but feature_dim={self.feature_dim}")

        lw = copy.deepcopy(loss_weights) if loss_weights is not None else LossWeights()

        x_seq, payload = self._extract_sequence(query_instance)
        T, D = x_seq.shape
        if T != self.sequence_length or D != self.feature_dim:
            raise ValueError(f"Expected sequence shape ({self.sequence_length}, {self.feature_dim}), got {tuple(x_seq.shape)}")

        X_orig_single = x_seq.unsqueeze(0).to(self.device)
        X_orig = X_orig_single.repeat(num_cfs, 1, 1)

        global_best_key = (float("inf"),) * 5
        global_best_X = None
        global_best_metrics = None
        global_best_restart = -1
        global_best_history = []

        for restart_idx in range(self.config.num_restarts):
            if restart_idx == 0:
                W0 = torch.zeros(num_cfs, self.num_basis, D, device=self.device)
            else:
                W0 = torch.randn(num_cfs, self.num_basis, D, device=self.device) * self.config.init_std
                if global_best_metrics is not None and "best_W" in global_best_metrics:
                    W0 = global_best_metrics["best_W"].detach().clone() + 0.5 * W0

            W = nn.Parameter(W0)

            adam_steps = min(self.config.adam_steps, self.config.max_iter)
            lbfgs_steps = min(self.config.lbfgs_steps, max(self.config.max_iter - adam_steps, 0))

            adam = optim.Adam([W], lr=self.config.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                adam,
                T_max=max(1, adam_steps),
                eta_min=self.config.eta_min,
            )

            rho = float(self.config.rho_init)
            patience = 0
            ever_valid = False
            local_best_key = (float("inf"),) * 5
            local_best_metrics = None
            history: List[Dict[str, float]] = []

            # ---------------- Adam stage ----------------
            for step in range(adam_steps):
                cleanup_scale = 1.0 if ever_valid else self.config.prevalid_cleanup_scale
                step_frac = float(step) / float(max(1, self.config.max_iter - 1))

                adam.zero_grad(set_to_none=True)
                metrics = self._compute_metrics(
                    X_orig=X_orig,
                    W=W,
                    schema=schema,
                    target=target,
                    num_cfs=num_cfs,
                    loss_weights=lw,
                    rho=rho,
                    cleanup_scale=cleanup_scale,
                    step_frac=step_frac,
                )
                metrics["total"].backward()
                torch.nn.utils.clip_grad_norm_([W], self.config.gradient_clip_norm)
                adam.step()
                scheduler.step()

                with torch.no_grad():
                    eval_metrics = self._compute_metrics(
                        X_orig=X_orig,
                        W=W,
                        schema=schema,
                        target=target,
                        num_cfs=num_cfs,
                        loss_weights=lw,
                        rho=rho,
                        cleanup_scale=1.0 if ever_valid else self.config.prevalid_cleanup_scale,
                        step_frac=step_frac,
                    )

                history.append({
                    "restart": float(restart_idx),
                    "step": float(step),
                    "total": float(eval_metrics["total"].detach().item()),
                    "validity": eval_metrics["validity"],
                    "proximity": eval_metrics["proximity"],
                    "coeff_sparsity": eval_metrics["coeff_sparsity"],
                    "group_sparsity": eval_metrics["group_sparsity"],
                    "smoothness": eval_metrics["smoothness"],
                    "diversity": eval_metrics["diversity"],
                    "pred_summary": eval_metrics["pred_summary"],
                    "is_valid": float(eval_metrics["is_valid"]),
                    "rho": float(rho),
                })

                key = self._selection_key(eval_metrics)
                if key < local_best_key:
                    local_best_key = key
                    local_best_metrics = {
                        **eval_metrics,
                        "best_W": W.detach().clone(),
                    }
                    patience = 0
                else:
                    patience += 1

                if eval_metrics["is_valid"]:
                    ever_valid = True

                if (step + 1) % 25 == 0 and not ever_valid:
                    rho = min(rho * self.config.rho_growth, self.config.rho_max)

                if ever_valid and patience >= self.config.early_stop_patience:
                    break

            # ---------------- LBFGS refinement ----------------
            if lbfgs_steps > 0:
                lbfgs = optim.LBFGS(
                    [W],
                    lr=0.5,
                    max_iter=lbfgs_steps,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    lbfgs.zero_grad()
                    metrics = self._compute_metrics(
                        X_orig=X_orig,
                        W=W,
                        schema=schema,
                        target=target,
                        num_cfs=num_cfs,
                        loss_weights=lw,
                        rho=rho,
                        cleanup_scale=1.0,
                        step_frac=1.0,
                    )
                    metrics["total"].backward()
                    torch.nn.utils.clip_grad_norm_([W], self.config.gradient_clip_norm)
                    return metrics["total"]

                lbfgs.step(closure)

                with torch.no_grad():
                    eval_metrics = self._compute_metrics(
                        X_orig=X_orig,
                        W=W,
                        schema=schema,
                        target=target,
                        num_cfs=num_cfs,
                        loss_weights=lw,
                        rho=rho,
                        cleanup_scale=1.0,
                        step_frac=1.0,
                    )
                history.append({
                    "restart": float(restart_idx),
                    "step": float(adam_steps),
                    "total": float(eval_metrics["total"].detach().item()),
                    "validity": eval_metrics["validity"],
                    "proximity": eval_metrics["proximity"],
                    "coeff_sparsity": eval_metrics["coeff_sparsity"],
                    "group_sparsity": eval_metrics["group_sparsity"],
                    "smoothness": eval_metrics["smoothness"],
                    "diversity": eval_metrics["diversity"],
                    "pred_summary": eval_metrics["pred_summary"],
                    "is_valid": float(eval_metrics["is_valid"]),
                    "rho": float(rho),
                })

                key = self._selection_key(eval_metrics)
                if key < local_best_key:
                    local_best_key = key
                    local_best_metrics = {
                        **eval_metrics,
                        "best_W": W.detach().clone(),
                    }

            if local_best_metrics is None:
                continue

            X_final, _ = self._apply_constraints(
                X_orig=X_orig,
                Delta=torch.einsum("tk,nkd->ntd", self.Phi, local_best_metrics["best_W"]),
                schema=schema,
                final_pass=True,
            )
            local_best_metrics["X_cf_final"] = X_final.detach()

            if local_best_key < global_best_key:
                global_best_key = local_best_key
                global_best_X = X_final.detach()
                global_best_metrics = local_best_metrics
                global_best_restart = restart_idx
                global_best_history = history

            if verbose:
                print(
                    f"[{self.basis_type}] restart={restart_idx} "
                    f"valid={local_best_metrics['is_valid']} "
                    f"validity={local_best_metrics['validity']:.6f} "
                    f"prox={local_best_metrics['proximity']:.6f} "
                    f"group={local_best_metrics['group_sparsity']:.6f} "
                    f"smooth={local_best_metrics['smoothness']:.6f}"
                )

        if global_best_X is None:
            raise RuntimeError("Counterfactual generation failed in all restarts.")

        info = {
            "basis_type": self.basis_type,
            "num_basis": self.num_basis,
            "best_restart": global_best_restart,
            "best_metrics": {
                "validity": float(global_best_metrics["validity"]),
                "proximity": float(global_best_metrics["proximity"]),
                "coeff_sparsity": float(global_best_metrics["coeff_sparsity"]),
                "group_sparsity": float(global_best_metrics["group_sparsity"]),
                "smoothness": float(global_best_metrics["smoothness"]),
                "diversity": float(global_best_metrics["diversity"]),
                "state_lock": float(global_best_metrics["state_lock"]),
                "pred_summary": float(global_best_metrics["pred_summary"]),
                "is_valid": bool(global_best_metrics["is_valid"]),
                "selection_key": tuple(float(x) for x in global_best_key),
                "total": float(global_best_metrics["total"].detach().item()),
            },
            "history": global_best_history,
        }
        self.last_history_ = global_best_history
        return global_best_X, info