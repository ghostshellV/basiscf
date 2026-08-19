import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, Dict, Any, List

from src.baselines.interface import CounterfactualExplainer

# ── Use the SAME shared losses that BasisGenerator uses ──────────────────────
from src.counterfactuals.losses import (
    proximity_loss,
    delta_sparsity_loss,
    smoothness_loss,
    validity_loss_regression,
)
from src.counterfactuals.core import (
    TargetSpec,
    TSFeatureSchema,
    LossWeights,
    GeneratorConfig,
    _to_tensor,
)
from src.evaluation.metrics import (
    CANONICAL_METRICS_SCHEMA_VERSION,
    get_canonical_history_key_contract,
    get_canonical_info_key_contract,
)

####
# COMTE: Counterfactual Explanations for Multivariate Time Series
#
# Paper: Ates, E., Aksar, B., Leung, V. J., & Coskun, A. K. (2021).
#        "Counterfactual Explanations for Multivariate Time Series."
#        2021 International Conference on Applied Artificial Intelligence (ICAPAI)
#
# Repository: https://github.com/peaclab/CoMTE
#
# Adapted here for **regression** tasks on time-series data.
#
# KEY CHANGE (v2): Now uses the shared loss functions from
# ``src.counterfactuals.losses`` (proximity_loss, sparsity_loss,
# smoothness_loss, validity_loss_regression) — the SAME functions
# used by all BasisGenerator variants (Fourier, B-Spline, etc.).
# This ensures a fair, apples-to-apples comparison.
####


class CoMTE(CounterfactualExplainer):
    """
    CoMTE: Counterfactual Explanations for Multivariate Time Series
    — adapted for **regression** tasks, using shared loss functions.

    Now uses the same ``proximity_loss``, ``sparsity_loss``,
    ``smoothness_loss``, and ``validity_loss_regression`` as
    ``BasisGenerator`` for fair comparisons.

    Supports two optimisation modes:
      - ``'standard'``: Gradient optimisation with all losses active
          from the start (validity + proximity + sparsity + smoothness).
      - ``'ts'``: Additionally adds temporal trend-preservation loss.

    Input convention
    ----------------
    Sequences are expected in **(T, D)** NumPy/Tensor format, matching the
    convention used by ``BasisGenerator`` and the project data-loaders.
    The model is called with shape **(1, T, D)**.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.05,
        max_iterations: int = 2000,
        mode: str = "standard",
        mutable_mask: Optional[np.ndarray] = None,
        min_vals: Optional[np.ndarray] = None,
        max_vals: Optional[np.ndarray] = None,
        mad_inv: Optional[np.ndarray] = None,
        gradient_clip_norm: float = 1.0,
        early_stop_patience: int = 300,
        device=None,
        verbose: bool = False,
    ):
        super().__init__(model)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.mode = mode
        self.mutable_mask = mutable_mask          # (D,) bool array or None
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.mad_inv = mad_inv
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stop_patience = early_stop_patience
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.last_history_: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query_instance,
        target: TargetSpec = None,
        target_value: Optional[float] = None,
        loss_weights: Optional[LossWeights] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Generate a counterfactual for *query_instance*.

        Parameters
        ----------
        query_instance : np.ndarray or torch.Tensor, shape ``(T, D)``
        target : TargetSpec
            Unified target specification (same as BasisGenerator).
        target_value : float, optional
            Shortcut — raw normalised target (used if ``target`` is None).
        loss_weights : LossWeights, optional
            Uses same weight structure as BasisGenerator.

        Returns
        -------
        cf_sample : np.ndarray of shape ``(T, D)`` or *None*
        cf_pred   : float — model prediction on the CF, or *None*
        """
        # Resolve target
        if target is not None:
            tv = target.target_value if target.target_value is not None else target_value
        else:
            tv = target_value
        if tv is None:
            raise ValueError("Provide target (TargetSpec) or target_value.")

        # Build a TargetSpec if only a raw float was given
        if target is None:
            target = TargetSpec(task_type="regression", target_value=float(tv))

        lw = loss_weights or LossWeights()

        if isinstance(query_instance, torch.Tensor):
            sample = query_instance.detach().cpu().numpy()
        else:
            sample = np.asarray(query_instance, dtype=np.float32)

        if self.mode == "ts":
            return self._generate_ts(sample, target, lw)
        return self._generate_standard(sample, target, lw)

    # ------------------------------------------------------------------
    # Standard optimisation (all losses active from start)
    # ------------------------------------------------------------------

    def _generate_standard(
        self,
        sample: np.ndarray,
        target: TargetSpec,
        lw: LossWeights,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Single-phase optimisation using shared losses:
          validity_loss_regression + proximity_loss + sparsity_loss + smoothness_loss
        All losses are active from iteration 0 (no free Phase-1).
        """
        x_orig = self._prepare_tensor(sample)                     # (1, T, D)
        mask = self._get_mutable_mask(sample.shape[-1])           # (D,) bool
        edit_mask = mask.float().view(1, 1, -1)                   # (1, 1, D)
        edit_mask_bool = edit_mask.bool()

        mad_inv_t = self._get_mad_inv_tensor(sample.shape[-1])

        # Pre-compute clamp bounds on device (avoid repeated CPU→GPU transfer)
        lo_t = _to_tensor(self.min_vals, self.device).view(1, 1, -1) if self.min_vals is not None else None
        hi_t = _to_tensor(self.max_vals, self.device).view(1, 1, -1) if self.max_vals is not None else None

        x_cf_free = nn.Parameter(x_orig.clone().detach())         # (1, T, D)
        optimizer = optim.Adam([x_cf_free], lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_iterations, eta_min=1e-5,
        )

        best_cf = None
        best_score = float("inf")
        best_total = float("inf")
        no_improve = 0
        history = []

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Lock immutable features
            x_eff = torch.where(edit_mask_bool, x_cf_free, x_orig)

            # Clamp to valid bounds (pre-computed tensors)
            if lo_t is not None:
                x_eff = torch.where(edit_mask_bool, torch.maximum(x_eff, lo_t), x_eff)
            if hi_t is not None:
                x_eff = torch.where(edit_mask_bool, torch.minimum(x_eff, hi_t), x_eff)

            Delta = x_eff - x_orig                                # (1, T, D)

            # Model forward
            pred = self.model(x_eff)
            y_model = pred.view(-1)                               # (1,) or (N,)

            # ── Shared losses (same functions as BasisGenerator) ─────────
            l_valid = validity_loss_regression(
                y_model,
                target_value=target.target_value,
                target_range=target.target_range,
            )
            l_prox = proximity_loss(Delta, mad_inv=mad_inv_t)
            l_sparse = delta_sparsity_loss(Delta)
            l_smooth = smoothness_loss(Delta)

            total_loss = (
                lw.validity   * l_valid
                + lw.proximity  * l_prox
                + lw.sparsity   * l_sparse
                + lw.smoothness * l_smooth
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [x_cf_free], max_norm=self.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()

            # ── Tracking ─────────────────────────────────────────────────
            with torch.no_grad():
                v_err = torch.abs(y_model - float(target.target_value)).mean().item()
                t_loss = total_loss.item()

                if (v_err < best_score) or (
                    abs(v_err - best_score) < 1e-12 and t_loss < best_total
                ):
                    best_score = v_err
                    best_total = t_loss
                    best_cf = x_eff.clone().detach()
                    no_improve = 0
                else:
                    no_improve += 1

                history.append({
                    "iter": iteration,
                    "total_loss": t_loss,
                    "l_valid": float(l_valid.item()),
                    "l_prox": float(l_prox.item()),
                    "l_sparse": float(l_sparse.item()),
                    "l_smooth": float(l_smooth.item()),
                    "validity_err": float(v_err),
                    "v_err": float(v_err),
                })

            if self.verbose and iteration % 200 == 0:
                print(
                    f"CoMTE iter {iteration:04d}: total={t_loss:.4f} | "
                    f"valid={l_valid.item():.4f} | prox={l_prox.item():.4f} | "
                    f"sparse={l_sparse.item():.4f} | smooth={l_smooth.item():.4f} | "
                    f"v_err={v_err:.4f}"
                )

            # Early stopping (same tolerance logic as BasisGenerator)
            if v_err < 1e-4:
                if self.verbose:
                    print(f"CoMTE: converged at iter {iteration}, err={v_err:.6f}")
                break
            if no_improve >= self.early_stop_patience:
                if self.verbose:
                    print(f"CoMTE: early stop (no improvement) at iter {iteration}")
                break

        self.last_history_ = history
        return self._finalise(best_cf, sample)

    # ------------------------------------------------------------------
    # CoMTE-TS: with temporal trend-preservation
    # ------------------------------------------------------------------

    def _generate_ts(
        self,
        sample: np.ndarray,
        target: TargetSpec,
        lw: LossWeights,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Same as standard but adds temporal trend-preservation loss:
        penalises change in second-order differences vs. original.
        """
        x_orig = self._prepare_tensor(sample)
        mask = self._get_mutable_mask(sample.shape[-1])
        edit_mask = mask.float().view(1, 1, -1)
        edit_mask_bool = edit_mask.bool()
        mad_inv_t = self._get_mad_inv_tensor(sample.shape[-1])

        # Pre-compute clamp bounds on device
        lo_t = _to_tensor(self.min_vals, self.device).view(1, 1, -1) if self.min_vals is not None else None
        hi_t = _to_tensor(self.max_vals, self.device).view(1, 1, -1) if self.max_vals is not None else None

        with torch.no_grad():
            orig_diff1 = x_orig[:, 1:, :] - x_orig[:, :-1, :]
            orig_diff2 = orig_diff1[:, 1:, :] - orig_diff1[:, :-1, :]

        x_cf_free = nn.Parameter(x_orig.clone().detach())
        optimizer = optim.Adam([x_cf_free], lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_iterations, eta_min=1e-5,
        )

        best_cf = None
        best_score = float("inf")
        best_total = float("inf")
        no_improve = 0
        history = []

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            x_eff = torch.where(edit_mask_bool, x_cf_free, x_orig)
            # Clamp to valid bounds (pre-computed tensors)
            if lo_t is not None:
                x_eff = torch.where(edit_mask_bool, torch.maximum(x_eff, lo_t), x_eff)
            if hi_t is not None:
                x_eff = torch.where(edit_mask_bool, torch.minimum(x_eff, hi_t), x_eff)
            Delta = x_eff - x_orig

            pred = self.model(x_eff)
            y_model = pred.view(-1)

            l_valid = validity_loss_regression(
                y_model,
                target_value=target.target_value,
                target_range=target.target_range,
            )
            l_prox = proximity_loss(Delta, mad_inv=mad_inv_t)
            l_sparse = delta_sparsity_loss(Delta)
            l_smooth = smoothness_loss(Delta)

            # Extra temporal trend-preservation
            cf_diff1 = x_eff[:, 1:, :] - x_eff[:, :-1, :]
            cf_diff2 = cf_diff1[:, 1:, :] - cf_diff1[:, :-1, :]
            l_temporal = torch.mean((cf_diff2 - orig_diff2) ** 2)

            total_loss = (
                lw.validity   * l_valid
                + lw.proximity  * l_prox
                + lw.sparsity   * l_sparse
                + lw.smoothness * l_smooth
                + 0.5 * l_temporal     # fixed weight for trend preservation
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [x_cf_free], max_norm=self.gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                v_err = torch.abs(y_model - float(target.target_value)).mean().item()
                t_loss = total_loss.item()

                if (v_err < best_score) or (
                    abs(v_err - best_score) < 1e-12 and t_loss < best_total
                ):
                    best_score = v_err
                    best_total = t_loss
                    best_cf = x_eff.clone().detach()
                    no_improve = 0
                else:
                    no_improve += 1

                history.append({
                    "iter": iteration,
                    "total_loss": float(t_loss),
                    "l_valid": float(l_valid.item()),
                    "l_prox": float(l_prox.item()),
                    "l_sparse": float(l_sparse.item()),
                    "l_smooth": float(l_smooth.item()),
                    "l_temporal": float(l_temporal.item()),
                    "validity_err": float(v_err),
                    "v_err": float(v_err),
                })

            if self.verbose and iteration % 200 == 0:
                print(
                    f"CoMTE-TS iter {iteration:04d}: total={t_loss:.4f} | "
                    f"valid={l_valid.item():.4f} | smooth={l_smooth.item():.4f} | "
                    f"temporal={l_temporal.item():.4f} | v_err={v_err:.4f}"
                )

            if v_err < 1e-4:
                if self.verbose:
                    print(f"CoMTE-TS: converged at iter {iteration}")
                break
            if no_improve >= self.early_stop_patience:
                if self.verbose:
                    print(f"CoMTE-TS: early stop at iter {iteration}")
                break

        self.last_history_ = history
        return self._finalise(best_cf, sample)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_tensor(self, sample: np.ndarray) -> torch.Tensor:
        """Convert ``(T, D)`` numpy to ``(1, T, D)`` float32 tensor."""
        x = torch.tensor(sample, dtype=torch.float32, device=self.device)
        if x.ndim == 2:
            x = x.unsqueeze(0)     # (T, D) → (1, T, D)
        elif x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # (T,) → (1, T, 1)
        return x

    def _get_mutable_mask(self, n_features: int) -> torch.Tensor:
        """Return a ``(D,)`` bool tensor: True = editable."""
        if self.mutable_mask is None:
            return torch.ones(n_features, dtype=torch.bool, device=self.device)
        m = torch.tensor(self.mutable_mask, dtype=torch.bool, device=self.device)
        if m.shape[0] != n_features:
            raise ValueError(
                f"mutable_mask length {m.shape[0]} != n_features {n_features}"
            )
        return m

    def _get_mad_inv_tensor(self, n_features: int) -> Optional[torch.Tensor]:
        """Convert stored mad_inv to device tensor, or None."""
        if self.mad_inv is None:
            return None
        return _to_tensor(self.mad_inv, self.device).view(-1)

    def _clamp_bounds(self, x_eff, x_orig, edit_mask):
        """Clamp editable features to [min_vals, max_vals]."""
        if self.min_vals is not None:
            lo = _to_tensor(self.min_vals, self.device).view(1, 1, -1)
            x_eff = torch.where(edit_mask.bool(), torch.maximum(x_eff, lo), x_eff)
        if self.max_vals is not None:
            hi = _to_tensor(self.max_vals, self.device).view(1, 1, -1)
            x_eff = torch.where(edit_mask.bool(), torch.minimum(x_eff, hi), x_eff)
        return x_eff

    def _finalise(
        self,
        best_cf: Optional[torch.Tensor],
        sample: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Convert best CF tensor back to numpy and get final prediction."""
        if best_cf is None:
            if self.verbose:
                print("CoMTE: no counterfactual found.")
            return None, None

        with torch.no_grad():
            final_pred = self.model(best_cf)
            cf_pred_val = final_pred.view(-1)[0].item()

        # (1, T, D) → (T, D)
        cf_sample = best_cf.squeeze(0).cpu().numpy()
        return cf_sample, cf_pred_val


class CoMTEGenerator(CounterfactualExplainer):
    """
    Wrapper that gives CoMTE the same high-level API as BasisGenerator.

    Notes
    -----
    - This wrapper standardises the interface only.
    - The underlying low-level CoMTE optimiser remains unchanged.
    - This wrapper supports regression TargetSpec with either:
        * target_value
        * target_range
    """

    def __init__(
        self,
        model: nn.Module,
        sequence_length: int,
        feature_dim: int,
        device: str = "cuda",
        config: Optional[GeneratorConfig] = None,
        mode: str = "standard",
        learning_rate: float = 0.05,
        max_iterations: int = 2000,
    ):
        super().__init__(model)
        self.T = sequence_length
        self.D = feature_dim
        self.device_str = device
        self.device = torch.device(device)

        self.config = config if config is not None else GeneratorConfig(device=device)
        self.mode = mode

        self._lr = getattr(self.config, "lr", learning_rate)
        self._max_iter = getattr(self.config, "max_iter", max_iterations)
        self._clip_norm = getattr(self.config, "gradient_clip_norm", 1.0)
        self._patience = getattr(self.config, "early_stop_patience", 300)

        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_mutable_mask(self, schema: Optional[TSFeatureSchema]) -> Optional[np.ndarray]:
        if schema is None:
            return None

        if schema.mutable_mask is not None:
            return np.asarray(schema.mutable_mask, dtype=bool)

        editable_roles = set(getattr(self.config, "editable_roles", ("action",)))
        allow_state_edits = bool(getattr(self.config, "allow_state_edits", False))

        mask = []
        for role in schema.roles:
            editable = role in editable_roles
            if role == "state":
                editable = editable or allow_state_edits
            if role in ("immutable", "context"):
                editable = False
            mask.append(editable)
        return np.asarray(mask, dtype=bool)

    @staticmethod
    def _target_error(pred: float, target: TargetSpec) -> float:
        if target.task_type != "regression":
            raise ValueError("Current CoMTE baseline wrapper supports regression targets only.")

        if target.target_range is not None:
            lo, hi = target.target_range
            if pred < lo:
                return float(lo - pred)
            if pred > hi:
                return float(pred - hi)
            return 0.0

        if target.target_value is None:
            raise ValueError("Regression target must define target_value or target_range.")
        return abs(float(pred) - float(target.target_value))

    @staticmethod
    def _is_valid(pred: float, target: TargetSpec) -> bool:
        return CoMTEGenerator._target_error(pred, target) <= 1e-12

    @staticmethod
    def _history_min(history, keys, default=float("nan")):
        vals = []
        for h in history:
            for k in keys:
                if k in h and h[k] is not None:
                    vals.append(float(h[k]))
                    break
        return min(vals) if vals else default

    def _selection_key(self, pred: float, history, target: TargetSpec):
        validity = self._target_error(pred, target)
        invalid = 0.0 if validity <= 1e-12 else 1.0
        proximity = self._history_min(history, ["l_prox", "proximity"], default=float("inf"))
        sparsity = self._history_min(history, ["l_sparse", "sparsity"], default=float("inf"))
        smoothness = self._history_min(history, ["l_smooth", "smoothness"], default=float("inf"))
        return (invalid, validity, proximity, sparsity, smoothness)

    def _predict_scalar(self, sample_np: np.ndarray) -> float:
        self.model.eval()
        x = torch.tensor(sample_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        return float(out.view(-1)[0].item())

    def _jitter_query(self, q_np: np.ndarray, schema: Optional[TSFeatureSchema]) -> np.ndarray:
        if schema is None:
            return q_np.copy()

        mutable_mask = self._build_mutable_mask(schema)
        noise = np.random.randn(*q_np.shape).astype(np.float32) * 0.002
        if mutable_mask is not None:
            noise = noise * mutable_mask.reshape(1, -1)

        q_new = q_np + noise

        if schema.min_vals is not None:
            q_new = np.maximum(q_new, np.asarray(schema.min_vals, dtype=np.float32).reshape(1, -1))
        if schema.max_vals is not None:
            q_new = np.minimum(q_new, np.asarray(schema.max_vals, dtype=np.float32).reshape(1, -1))

        return q_new.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query_instance: torch.Tensor,
        target: TargetSpec,
        schema: Optional[TSFeatureSchema] = None,
        num_cfs: int = 1,
        loss_weights: Optional[LossWeights] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if target.task_type != "regression":
            raise ValueError("Current CoMTE baseline wrapper supports regression targets only.")

        lw = loss_weights or LossWeights()

        mutable_mask = self._build_mutable_mask(schema)
        mad_inv = np.asarray(schema.mad_inv, dtype=np.float32) if (schema is not None and schema.mad_inv is not None) else None
        min_vals = np.asarray(schema.min_vals, dtype=np.float32) if (schema is not None and schema.min_vals is not None) else None
        max_vals = np.asarray(schema.max_vals, dtype=np.float32) if (schema is not None and schema.max_vals is not None) else None

        comte = CoMTE(
            model=self.model,
            learning_rate=self._lr,
            max_iterations=self._max_iter,
            mode=self.mode,
            mutable_mask=mutable_mask,
            min_vals=min_vals,
            max_vals=max_vals,
            mad_inv=mad_inv,
            gradient_clip_norm=self._clip_norm,
            early_stop_patience=self._patience,
            device=self.device_str,
            verbose=verbose,
        )

        q_np = (
            query_instance.detach().cpu().numpy()
            if isinstance(query_instance, torch.Tensor)
            else np.asarray(query_instance, dtype=np.float32)
        )

        cf_list: List[torch.Tensor] = []
        cf_preds: List[float] = []
        histories: List[List[Dict[str, Any]]] = []
        run_keys: List[Tuple[float, ...]] = []

        for run_i in range(num_cfs):
            q_input = q_np.astype(np.float32) if run_i == 0 else self._jitter_query(q_np, schema)

            cf_np, cf_pred = comte.generate(
                query_instance=q_input,
                target=target,
                loss_weights=lw,
            )

            if cf_np is None:
                cf_np = q_np.copy()
                cf_pred = self._predict_scalar(cf_np)

            hist = list(comte.last_history_)
            key = self._selection_key(float(cf_pred), hist, target)

            cf_list.append(torch.tensor(cf_np, dtype=torch.float32, device=self.device))
            cf_preds.append(float(cf_pred))
            histories.append(hist)
            run_keys.append(key)

        best_idx = min(range(len(run_keys)), key=lambda i: run_keys[i])
        cfs = torch.stack(cf_list, dim=0)

        best_history = histories[best_idx]
        best_pred = cf_preds[best_idx]
        best_error = self._target_error(best_pred, target)

        info = {
            "baseline": "comte",
            "best_restart": int(best_idx),
            "history": best_history,
            "history_by_run": histories,
            "cf_pred": best_pred,
            "all_cf_preds": cf_preds,
            "error": best_error,
            "success": self._is_valid(best_pred, target),
            "feature_names": schema.feature_names if schema is not None else None,
            "roles": schema.roles if schema is not None else None,
            "best_metrics": {
                "validity": self._history_min(best_history, ["validity_err", "v_err", "l_valid"], default=best_error),
                "proximity": self._history_min(best_history, ["l_prox", "proximity"]),
                "coeff_sparsity": float("nan"),
                "group_sparsity": float("nan"),
                "smoothness": self._history_min(best_history, ["l_smooth", "smoothness"]),
                "diversity": 0.0,
                "state_lock": 0.0,
                "pred_summary": best_pred,
                "is_valid": self._is_valid(best_pred, target),
                "selection_key": run_keys[best_idx],
                "total": self._history_min(best_history, ["total_loss", "total"]),
            },
        }
        return cfs, info

# ──────────────────────────────────────────────────────────────────────────────
# CoMTEGenerator — drop-in replacement for BasisGenerator
#
# Wraps ``CoMTE`` and exposes the *same* ``generate()`` signature as
# ``BasisGenerator`` so it can be used side-by-side in comparison loops.
#
# Now uses TargetSpec, LossWeights, TSFeatureSchema from core.py
# and the shared loss functions — ensuring identical evaluation.
# ──────────────────────────────────────────────────────────────────────────────

# class CoMTEGenerator:
#     """
#     Wrapper that gives ``CoMTE`` the same public API as ``BasisGenerator``.

#     Uses TargetSpec / LossWeights / TSFeatureSchema identically to
#     BasisGenerator, so targets and evaluation are exactly comparable.

#     Usage::

#         comte_gen = CoMTEGenerator(
#             model=model,
#             sequence_length=seq_len,
#             feature_dim=n_features,
#             device=str(device),
#             config=gen_config,
#             mode='standard',
#         )

#         cfs, info = comte_gen.generate(
#             query_instance=query_tensor,   # (T, D)
#             target=target_spec,            # TargetSpec(task_type='regression', target_value=…)
#             schema=feature_schema,         # TSFeatureSchema
#             num_cfs=3,
#             loss_weights=loss_weights,     # LossWeights
#             verbose=False,
#         )
#     """

#     def __init__(
#         self,
#         model: nn.Module,
#         sequence_length: int,
#         feature_dim: int,
#         device: str = "cuda",
#         mode: str = "standard",
#         config: Optional[GeneratorConfig] = None,
#         learning_rate: float = 0.05,
#         max_iterations: int = 2000,
#     ):
#         self.T = sequence_length
#         self.D = feature_dim
#         self.device_str = device
#         self.mode = mode
#         _device = torch.device(device)

#         # Override from GeneratorConfig if provided
#         if config is not None:
#             lr = getattr(config, "lr", learning_rate)
#             max_iter = getattr(config, "max_iter", max_iterations)
#             clip_norm = getattr(config, "gradient_clip_norm", 1.0)
#             patience = getattr(config, "early_stop_patience", 300)
#         else:
#             lr = learning_rate
#             max_iter = max_iterations
#             clip_norm = 1.0
#             patience = 300

#         self._lr = lr
#         self._max_iter = max_iter
#         self._clip_norm = clip_norm
#         self._patience = patience

#         self.model = model
#         self.model.to(_device)
#         self.model.eval()
#         for p in self.model.parameters():
#             p.requires_grad = False

#     # ------------------------------------------------------------------

#     def generate(
#         self,
#         query_instance: torch.Tensor,
#         target: TargetSpec,
#         schema: Optional[TSFeatureSchema] = None,
#         num_cfs: int = 1,
#         loss_weights: Optional[LossWeights] = None,
#         verbose: bool = False,
#     ) -> Tuple[torch.Tensor, Dict[str, Any]]:
#         """
#         Generate *num_cfs* counterfactuals using CoMTE.

#         Uses the same TargetSpec and LossWeights as BasisGenerator.

#         Returns
#         -------
#         cfs : torch.Tensor of shape ``(num_cfs, T, D)``
#         info : dict with keys ``history``, ``cf_pred``, ``error``, ``success``
#         """
#         lw = loss_weights or LossWeights()

#         # Build mutable mask + metadata from schema
#         mutable_mask = None
#         mad_inv = None
#         min_vals = None
#         max_vals = None
#         if schema is not None:
#             mutable_mask = self._build_mutable_mask(schema)
#             if schema.mad_inv is not None:
#                 mad_inv = np.asarray(schema.mad_inv, dtype=np.float32)
#             if schema.min_vals is not None:
#                 min_vals = np.asarray(schema.min_vals, dtype=np.float32)
#             if schema.max_vals is not None:
#                 max_vals = np.asarray(schema.max_vals, dtype=np.float32)

#         # Build low-level CoMTE instance with proper constraints
#         comte = CoMTE(
#             model=self.model,
#             learning_rate=self._lr,
#             max_iterations=self._max_iter,
#             mode=self.mode,
#             mutable_mask=mutable_mask,
#             min_vals=min_vals,
#             max_vals=max_vals,
#             mad_inv=mad_inv,
#             gradient_clip_norm=self._clip_norm,
#             early_stop_patience=self._patience,
#             device=self.device_str,
#             verbose=verbose,
#         )

#         # Query as numpy (T, D)
#         q_np = query_instance.detach().cpu().numpy() \
#             if isinstance(query_instance, torch.Tensor) else np.asarray(query_instance)

#         cf_list: List[torch.Tensor] = []
#         cf_preds: List[float] = []
#         all_history: List[Any] = []

#         for run_i in range(num_cfs):
#             # Small noise for diversity in subsequent runs
#             if run_i > 0:
#                 noise = np.random.randn(*q_np.shape) * 0.002
#                 q_input = (q_np + noise).astype(np.float32)
#             else:
#                 q_input = q_np.astype(np.float32)

#             cf_np, cf_pred = comte.generate(
#                 query_instance=q_input,
#                 target=target,
#                 loss_weights=lw,
#             )

#             if cf_np is None:
#                 cf_np = q_np.copy()
#                 cf_pred = self._predict_scalar(q_np)

#             cf_list.append(torch.tensor(cf_np, dtype=torch.float32))
#             cf_preds.append(float(cf_pred) if cf_pred is not None else float('nan'))
#             all_history.append(comte.last_history_)

#         cfs = torch.stack(cf_list, dim=0)   # (num_cfs, T, D)

#         primary_error = abs(cf_preds[0] - target.target_value) if cf_preds else float('nan')
#         best_total_loss = float("inf")
#         best_validity_err = float("inf")
#         if all_history and all_history[0]:
#             best_total_loss = min(h.get("total_loss", float("inf")) for h in all_history[0])
#             best_validity_err = min(h.get("validity_err", float("inf")) for h in all_history[0])

#         info = {
#             'metrics_schema_version': CANONICAL_METRICS_SCHEMA_VERSION,
#             'history_key_contract': get_canonical_history_key_contract(),
#             'info_key_contract': get_canonical_info_key_contract(),
#             'best_validity_err': best_validity_err,
#             'best_total_loss': best_total_loss,
#             'history': all_history[0] if all_history else [],
#             'history_by_run': all_history,
#             'cf_pred': cf_preds[0] if cf_preds else None,
#             'error': primary_error,
#             'success': primary_error < 0.05,
#             'all_cf_preds': cf_preds,
#             'weights': None,
#             'feature_names': schema.feature_names if schema is not None else None,
#             'roles': schema.roles if schema is not None else None,
#         }
#         return cfs, info

#     # ------------------------------------------------------------------
#     # Helpers
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _build_mutable_mask(schema: TSFeatureSchema) -> np.ndarray:
#         """
#         Return bool array (D,) from TSFeatureSchema.
#         'action' and 'state' are editable; 'immutable' and 'context' are locked.
#         """
#         editable_roles = {"action", "state"}
#         return np.array(
#             [r in editable_roles for r in schema.roles], dtype=bool
#         )

#     def _predict_scalar(self, sample_np: np.ndarray) -> float:
#         """Run model on a single (T, D) sample and return scalar prediction."""
#         self.model.eval()
#         dev = next(self.model.parameters()).device
#         x = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0).to(dev)
#         with torch.no_grad():
#             out = self.model(x)
#         return out.view(-1)[0].item()
