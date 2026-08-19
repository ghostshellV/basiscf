import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.EarlyStopping import EarlyStopping


# ── Metric helpers ───────────────────────────────────────────────────────────
# A custom metric function signature:  (y_true: ndarray, y_pred: ndarray) -> float
CustomMetricFn = Callable[[np.ndarray, np.ndarray], float]


class RegressionMetrics:
    """Generic regression metrics — dataset-agnostic."""

    @staticmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"mse": float(mse), "rmse": rmse, "mae": mae, "r2": r2}


class NASAScore:
    """CMAPSS-specific asymmetric scoring function (for backward compat)."""

    @staticmethod
    def __call__(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return NASAScore.compute(y_true, y_pred)

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        diff = y_pred - y_true
        return float(np.sum(np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)))


# Backward-compatible alias so existing code that imports RULMetrics still works.
class RULMetrics:
    nasa_score = staticmethod(NASAScore.compute)

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        metrics = RegressionMetrics.calculate(y_true, y_pred)
        metrics["nasa_score"] = NASAScore.compute(y_true, y_pred)
        return metrics


@dataclass
class TrainingConfig:
    # core
    num_epochs: int = 180
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    gradient_clip_value: float = 1.0
    # saving/logging
    dataset_name: str = "generic"
    save_path: str = "../outputs/saved_models"
    model_name: str = "model"
    save_every_n_epochs: int = 10
    print_every_n_epochs: int = 5

    # early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    early_stopping_start_epoch: int = 0

    # scheduler (ReduceLROnPlateau)
    scheduler_factor: float = 0.6
    scheduler_patience: int = 30
    scheduler_mode: str = "min"
    scheduler_min_lr: float = 1e-6
    scheduler_metric: str = "loss"  # metric fed to ReduceLROnPlateau.step()

    # BEST model selection + early stop monitor (must exist in val_metrics)
    monitor_metric: str = "loss"    # "loss", "rmse", "nasa_score", etc.
    monitor_mode: Optional[str] = None  # if None -> inferred

    # Extra dataset-specific metrics.
    # Mapping of name -> callable(y_true, y_pred) -> float.
    # These are computed alongside the generic regression metrics.
    # Example:  {"nasa_score": NASAScore.compute}
    custom_metrics: Dict[str, CustomMetricFn] = field(default_factory=dict)

    # Which extra metric name to show on the tqdm progress bar (optional).
    # Set to None to only show train_loss / val_loss.
    progress_bar_extra_metric: Optional[str] = None


class Trainer:
    def __init__(self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Optional[TrainingConfig] = None,
        preprocess: Optional[Dict[str, Any]] = None):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or TrainingConfig()
        self.preprocess = preprocess

        os.makedirs(self.config.save_path, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=self.config.scheduler_mode,
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.scheduler_min_lr,
        )

        # Build history keys dynamically: base metrics + any custom ones
        self._base_metric_names = ["loss", "rmse", "mae", "r2"]
        self._custom_metric_names = sorted(self.config.custom_metrics.keys())
        self._all_val_metric_names = self._base_metric_names + self._custom_metric_names

        self.history: Dict[str, List[float]] = {"train_loss": [], "learning_rate": []}
        for m in self._all_val_metric_names:
            self.history[f"val_{m}"] = []

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0

        self._setup_early_stopping()

    # ---------- helpers ---------- #

    def _infer_monitor_mode(self) -> str:
        if self.config.monitor_mode in ("min", "max"):
            return self.config.monitor_mode

        m = self.config.monitor_metric.lower()
        # lower-is-better metrics
        if m in ("loss", "mse", "rmse", "mae", "nasa_score"):
            return "min"
        # higher-is-better metrics
        if m in ("r2", "r2_score", "accuracy", "f1", "auc"):
            return "max"
        return "min"

    def _is_improvement(self, current: float) -> bool:
        mode = self._infer_monitor_mode()
        if self.best_score is None:
            return True
        if mode == "min":
            return current < self.best_score
        return current > self.best_score

    @staticmethod
    def _jsonable(x: Any) -> Any:
        # make preprocess safe for json
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        if isinstance(x, dict):
            return {k: Trainer._jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [Trainer._jsonable(v) for v in x]
        return x

    def _paths_for(self, tag: str) -> Tuple[str, str, str]:
        weights_path = os.path.join(self.config.save_path, f"{self.config.model_name}_{tag}.pth")
        ckpt_path = os.path.join(self.config.save_path, f"{self.config.model_name}_{tag}.ckpt")
        meta_path = os.path.join(self.config.save_path, f"{self.config.model_name}_{tag}_meta.json")
        return weights_path, ckpt_path, meta_path

    # ---------- early stopping ---------- #

    def _setup_early_stopping(self) -> None:
        # EarlyStopping will write its own checkpoint file, but we DO NOT rely on it for "best".
        # We keep it just to stop training.
        es_path = os.path.join(self.config.save_path, f"{self.config.model_name}_earlystop.pth")

        # Works with both “simple” EarlyStopping and newer variants
        try:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                verbose=True,
                path=es_path,
                start_epoch=self.config.early_stopping_start_epoch,
                min_delta=self.config.early_stopping_min_delta,
                mode=self._infer_monitor_mode(),
            )
        except TypeError:
            # fallback for older EarlyStopping signatures
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                verbose=True,
                path=es_path,
                start_epoch=self.config.early_stopping_start_epoch,
            )

    # ---------- core loops ---------- #

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]", leave=False)
        for x, y in pbar:
            x = x.to(self.device).float()
            y = y.to(self.device).float().view(-1)

            self.optimizer.zero_grad(set_to_none=True)
            yhat = self.model(x).view(-1)
            loss = self.criterion(yhat, y)

            loss.backward()
            if self.config.gradient_clip_value and self.config.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

            self.optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        pbar.close()
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        preds: List[float] = []
        trues: List[float] = []

        num_batches = len(self.val_loader)
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config.num_epochs} [Val]  ", leave=False)

        for x, y in pbar:
            x = x.to(self.device).float()
            y = y.to(self.device).float().view(-1)

            yhat = self.model(x).view(-1)
            loss = self.criterion(yhat, y)

            total_loss += float(loss.item())
            preds.extend(yhat.detach().cpu().numpy().tolist())
            trues.extend(y.detach().cpu().numpy().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        pbar.close()

        y_true = np.array(trues, dtype=np.float32)
        y_pred = np.array(preds, dtype=np.float32)

        # Generic regression metrics
        metrics = RegressionMetrics.calculate(y_true, y_pred)
        metrics["loss"] = float(total_loss / max(1, num_batches))

        # Dataset-specific custom metrics
        for name, fn in self.config.custom_metrics.items():
            metrics[name] = float(fn(y_true, y_pred))

        return metrics

    # ---------- saving ---------- #

    def save_history(self) -> None:
        path = os.path.join(self.config.save_path, f"{self.config.model_name}_history.json")
        serial = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        with open(path, "w") as f:
            json.dump(serial, f, indent=2)

    def save_checkpoint(self, epoch: int, tag: str) -> None:
        """
        Writes:
        - <model_name>_<tag>.pth   : pure model.state_dict()  (for easy inference)
        - <model_name>_<tag>.ckpt  : full checkpoint dict     (for resume/repro)
        - <model_name>_<tag>_meta.json : tiny human-readable metadata
        """
        weights_path, ckpt_path, meta_path = self._paths_for(tag)

        # (A) weights only (.pth) — best for inference
        torch.save(self.model.state_dict(), weights_path)

        # (B) full ckpt (.ckpt)
        preprocess = self._jsonable(self.preprocess) if self.preprocess is not None else None

        ckpt = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_score": float(self.best_score) if self.best_score is not None else None,
            "best_epoch": int(self.best_epoch),
            "history": self.history,
            "config": {k: v for k, v in self.config.__dict__.items() if k not in ("custom_metrics",)},
            "preprocess": preprocess,
        }
        torch.save(ckpt, ckpt_path)

        # (C) tiny meta json (optional)
        try:
            meta = {
                "epoch": int(epoch),
                "best_epoch": int(self.best_epoch),
                "best_score": float(self.best_score) if self.best_score is not None else None,
                "monitor_metric": self.config.monitor_metric,
                "monitor_mode": self._infer_monitor_mode(),
                "weights_file": os.path.basename(weights_path),
                "ckpt_file": os.path.basename(ckpt_path),
                "n_features": len(preprocess.get("feature_cols", [])) if isinstance(preprocess, dict) else None,
                "sequence_length": preprocess.get("sequence_length") if isinstance(preprocess, dict) else None,
                "max_rul": preprocess.get("max_rul") if isinstance(preprocess, dict) else None,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    def load_checkpoint(self, path: str) -> None:
        """
        Supports:
        - weights-only .pth  (state_dict)
        - full .ckpt         (dict with model_state_dict, optimizer_state_dict, ...)
        """
        obj = torch.load(path, map_location=self.device)
        if isinstance(obj, dict) and "model_state_dict" in obj:
            self.model.load_state_dict(obj["model_state_dict"])
            if "optimizer_state_dict" in obj and obj["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(obj["optimizer_state_dict"])
            if "scheduler_state_dict" in obj and obj["scheduler_state_dict"] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(obj["scheduler_state_dict"])
            self.best_score = obj.get("best_score", self.best_score)
            self.best_epoch = obj.get("best_epoch", self.best_epoch)
            if "history" in obj and obj["history"] is not None:
                self.history = obj["history"]
            self.preprocess = obj.get("preprocess", self.preprocess)
        else:
            # assume weights-only state_dict
            self.model.load_state_dict(obj)

    # ---------- train ---------- #

    def _print_epoch_summary(self, epoch: int, train_loss: float, val_metrics: Dict[str, float]) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"Epoch [{epoch}/{self.config.num_epochs}] Summary")
        tqdm.write(f"{'='*80}")
        tqdm.write(f"  Train Loss:     {train_loss:.6f}")
        # Print all validation metrics dynamically
        for name in self._all_val_metric_names:
            if name in val_metrics:
                label = f"Val {name}".ljust(16)
                tqdm.write(f"  {label}: {val_metrics[name]:.6f}")
        tqdm.write(f"  LR:             {lr:.8f}")
        tqdm.write(f"  Monitor:        {self.config.monitor_metric} ({self._infer_monitor_mode()})")
        tqdm.write(f"{'='*80}\n")

    def train(self) -> Dict[str, List[float]]:
        print(f"\n{'='*80}")
        print(f"Training Model: {self.config.model_name}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  Epochs:        {self.config.num_epochs}")
        print(f"  LR:            {self.config.learning_rate}")
        print(f"  Batch Size:    {self.config.batch_size}")
        print(f"  Device:        {self.device}")
        print(f"  Save Path:     {self.config.save_path}")
        print(f"  Monitor:       {self.config.monitor_metric} ({self._infer_monitor_mode()})")
        print(f"{'='*80}\n")

        overall = tqdm(range(1, self.config.num_epochs + 1), desc="Overall Progress", position=0)

        for epoch in overall:
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            # Scheduler: ReduceLROnPlateau expects a metric value after validation. 
            sched_key = self.config.scheduler_metric
            if self.scheduler is not None:
                metric_for_sched = float(val_metrics.get(sched_key, val_metrics["loss"]))
                self.scheduler.step(metric_for_sched)

            lr = self.optimizer.param_groups[0]["lr"]

            # history — dynamic based on available metrics
            self.history["train_loss"].append(float(train_loss))
            for m in self._all_val_metric_names:
                if m in val_metrics:
                    self.history[f"val_{m}"].append(float(val_metrics[m]))
            self.history["learning_rate"].append(float(lr))

            # progress bar — always show train/val loss + optional extra metric
            postfix = {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_metrics['loss']:.4f}",
                "lr": f"{lr:.2e}",
            }
            extra = self.config.progress_bar_extra_metric
            if extra and extra in val_metrics:
                postfix[extra] = f"{val_metrics[extra]:.2f}"
            overall.set_postfix(**postfix)

            # print
            if epoch % self.config.print_every_n_epochs == 0 or epoch == 1:
                self._print_epoch_summary(epoch, train_loss, val_metrics)

            # best + save (same metric as early stop)
            monitor_key = self.config.monitor_metric
            if monitor_key not in val_metrics:
                raise KeyError(f"monitor_metric='{monitor_key}' not found in val_metrics keys={list(val_metrics.keys())}")

            monitor_value = float(val_metrics[monitor_key])

            if self._is_improvement(monitor_value):
                self.best_score = monitor_value
                self.best_epoch = epoch
                self.save_checkpoint(epoch, tag="best")

            # early stopping check (same monitor)
            self.early_stopping(monitor_value, self.model, epoch)
            if getattr(self.early_stopping, "early_stop", False):
                tqdm.write(f"\n⚠ Early stopping at epoch {epoch}")
                break

            if epoch % self.config.save_every_n_epochs == 0:
                self.save_history()
                self.save_checkpoint(epoch, tag="last")

        overall.close()

        # Save final "last" and history
        self.save_checkpoint(epoch, tag="last")
        self.save_history()

        # Load best weights into model for downstream use
        best_weights, _, _ = self._paths_for("best")
        if os.path.exists(best_weights):
            # weights-only is the correct inference path. :contentReference[oaicite:6]{index=6}
            self.model.load_state_dict(torch.load(best_weights, map_location=self.device))

        print(f"\n{'='*80}")
        print("Training Completed!")
        print(f"{'='*80}")
        print(f"  Best Epoch:      {self.best_epoch}")
        print(f"  Best({self.config.monitor_metric}): {self.best_score}")
        print(f"  Saved to:        {self.config.save_path}")
        print(f"{'='*80}\n")

        return self.history

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        preds: List[float] = []
        trues: List[float] = []

        for x, y in tqdm(test_loader, desc="Evaluating", leave=True):
            x = x.to(self.device).float()
            y = y.to(self.device).float().view(-1)
            yhat = self.model(x).view(-1)

            loss = self.criterion(yhat, y)
            total_loss += float(loss.item())

            preds.extend(yhat.detach().cpu().numpy().tolist())
            trues.extend(y.detach().cpu().numpy().tolist())

        y_true = np.array(trues, dtype=np.float32)
        y_pred = np.array(preds, dtype=np.float32)

        metrics = RegressionMetrics.calculate(y_true, y_pred)
        metrics["loss"] = float(total_loss / max(1, len(test_loader)))

        for name, fn in self.config.custom_metrics.items():
            metrics[name] = float(fn(y_true, y_pred))

        return metrics, y_pred, y_true
