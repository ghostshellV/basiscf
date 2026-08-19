from __future__ import annotations

import contextlib
import inspect
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


Array = np.ndarray
Batch = Any
CustomMetricFn = Callable[[Array, Array], float]
MetricTransformFn = Callable[[Array, Array, Optional[List[Any]]], Tuple[Array, Array]]


class RegressionMetrics:
    @staticmethod
    def calculate(y_true: Array, y_pred: Array) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        out = {"mse": float(mse), "rmse": rmse, "mae": mae}
        try:
            out["r2"] = float(r2_score(y_true, y_pred))
        except Exception:
            out["r2"] = float("nan")
        return out


class ClassificationMetrics:
    @staticmethod
    def calculate_binary(y_true: Array, y_prob: Array, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(int).reshape(-1)
        y_prob = np.asarray(y_prob).reshape(-1)
        y_pred = (y_prob >= threshold).astype(int)

        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            out["auroc"] = float("nan")
        return out

    @staticmethod
    def calculate_multiclass(y_true: Array, logits_or_prob: Array) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(int).reshape(-1)
        scores = np.asarray(logits_or_prob)
        y_pred = scores.argmax(axis=1)
        out = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        try:
            out["auroc_ovr"] = float(roc_auc_score(y_true, scores, multi_class="ovr"))
        except Exception:
            out["auroc_ovr"] = float("nan")
        return out

    @staticmethod
    def calculate_multilabel(y_true: Array, y_prob: Array, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)
        out = {
            "accuracy": float((y_pred == y_true).mean()),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        try:
            out["auroc_macro"] = float(roc_auc_score(y_true, y_prob, average="macro"))
        except Exception:
            out["auroc_macro"] = float("nan")
        return out


class NASAScore:
    @staticmethod
    def __call__(y_true: Array, y_pred: Array) -> float:
        return NASAScore.compute(y_true, y_pred)

    @staticmethod
    def compute(y_true: Array, y_pred: Array) -> float:
        diff = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
        return float(np.sum(np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)))


class RULMetrics:
    nasa_score = staticmethod(NASAScore.compute)

    @staticmethod
    def calculate_all_metrics(y_true: Array, y_pred: Array) -> Dict[str, float]:
        metrics = RegressionMetrics.calculate(y_true, y_pred)
        metrics["nasa_score"] = NASAScore.compute(y_true, y_pred)
        return metrics


@dataclass
class EarlyStoppingState:
    patience: int = 20
    min_delta: float = 0.0
    mode: str = "min"
    start_epoch: int = 0
    counter: int = 0
    best_score: Optional[float] = None
    early_stop: bool = False

    def step(self, metric_value: float, epoch: int) -> bool:
        if epoch < self.start_epoch:
            return False

        if self.best_score is None:
            self.best_score = metric_value
            return False

        improved = (
            metric_value < (self.best_score - self.min_delta)
            if self.mode == "min"
            else metric_value > (self.best_score + self.min_delta)
        )
        if improved:
            self.best_score = metric_value
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


@dataclass
class TrainingConfig:
    # Task
    task_type: str = "regression"  # regression | binary | multiclass | multilabel
    num_classes: Optional[int] = None
    multilabel_threshold: float = 0.5

    # Optimisation
    num_epochs: int = 180
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    gradient_clip_value: float = 1.0
    accumulation_steps: int = 1
    optimizer_name: str = "adamw"  # adam | adamw | sgd | rmsprop
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Loss
    loss_name: Optional[str] = None  # None -> inferred from task_type
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Scheduler
    scheduler_name: Optional[str] = "reduce_on_plateau"  # reduce_on_plateau | onecycle | cosine | none
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    scheduler_metric: str = "loss"

    # AMP / runtime
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16 | bfloat16
    use_inference_mode: bool = True
    non_blocking: bool = True

    # Saving / logging
    dataset_name: str = "generic"
    save_path: str = "../outputs/saved_models"
    model_name: str = "model"
    save_every_n_epochs: int = 10
    print_every_n_epochs: int = 5

    # Monitoring / early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    early_stopping_start_epoch: int = 0
    monitor_metric: str = "loss"
    monitor_mode: Optional[str] = None

    # Batch / output parsing
    batch_input_keys: Optional[List[str]] = None
    batch_target_key: Optional[str] = None
    batch_meta_key: Optional[str] = None
    output_key: Optional[str] = None
    output_index: int = 0

    # Metric handling
    compute_train_metrics: bool = False
    custom_metrics: Dict[str, CustomMetricFn] = field(default_factory=dict)
    progress_bar_extra_metric: Optional[str] = None
    metric_transform: Optional[MetricTransformFn] = None

    # User extensibility
    criterion_fn: Optional[nn.Module] = None
    optimizer_factory: Optional[Callable[[Iterable[torch.nn.Parameter]], optim.Optimizer]] = None
    scheduler_factory: Optional[Callable[[optim.Optimizer], Any]] = None




def make_affine_target_metric_transform(target_min: Union[float, Array], target_range: Union[float, Array]) -> MetricTransformFn:
    """
    Returns a metric transform that inverse-transforms targets/predictions before metric computation.
    Useful for generic normalised regression targets.
    """
    target_min_arr = np.asarray(target_min, dtype=np.float32)
    target_range_arr = np.asarray(target_range, dtype=np.float32)

    def _transform(y_true: Array, y_pred: Array, meta: Optional[List[Any]] = None) -> Tuple[Array, Array]:
        yt = np.asarray(y_true, dtype=np.float32) * target_range_arr + target_min_arr
        yp = np.asarray(y_pred, dtype=np.float32) * target_range_arr + target_min_arr
        return yt, yp

    return _transform


def make_cmapss_metric_transform(preprocess: Dict[str, Any]) -> MetricTransformFn:
    """
    Convenience helper for CMAPSS/RUL-style checkpoints whose preprocess dict stores
    either global_min/global_range or max_rul.
    """
    if preprocess is None:
        raise ValueError("preprocess must not be None")

    if "global_min" in preprocess and "global_range" in preprocess:
        global_min = np.asarray(preprocess["global_min"], dtype=np.float32)
        global_range = np.asarray(preprocess["global_range"], dtype=np.float32)
        target_min = float(global_min[-1])
        target_range = float(global_range[-1])
        return make_affine_target_metric_transform(target_min, target_range)

    target_min = 0.0
    target_range = float(preprocess.get("max_rul", 1.0))
    return make_affine_target_metric_transform(target_min, target_range)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Optional[TrainingConfig] = None,
        preprocess: Optional[Dict[str, Any]] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or TrainingConfig()
        self.preprocess = preprocess

        os.makedirs(self.config.save_path, exist_ok=True)

        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()

        self._base_metric_names = self._infer_metric_names()
        self._custom_metric_names = sorted(self.config.custom_metrics.keys())
        self._all_val_metric_names = self._base_metric_names + self._custom_metric_names

        self.history: Dict[str, List[float]] = {"train_loss": [], "learning_rate": []}
        if self.config.compute_train_metrics:
            for m in self._base_metric_names:
                self.history[f"train_{m}"] = []
        for m in self._all_val_metric_names:
            self.history[f"val_{m}"] = []

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.early_stopping = EarlyStoppingState(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode=self._infer_monitor_mode(),
            start_epoch=self.config.early_stopping_start_epoch,
        )

    # --------------------------- builders --------------------------- #

    def _resolve_amp_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.config.amp_dtype.lower() == "bfloat16" else torch.float16

    def _amp_enabled(self) -> bool:
        return bool(self.config.use_amp and self.device.type == "cuda")

    def _autocast_context(self):
        if not self._amp_enabled():
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self._resolve_amp_dtype())

    def _build_scaler(self):
        enabled = self._amp_enabled()
        try:
            return torch.amp.GradScaler(self.device.type, enabled=enabled)
        except Exception:
            if self.device.type == "cuda":
                return torch.cuda.amp.GradScaler(enabled=enabled)
            return None

    def _build_optimizer(self):
        if self.config.optimizer_factory is not None:
            return self.config.optimizer_factory(self.model.parameters())

        name = self.config.optimizer_name.lower()
        kwargs = dict(self.config.optimizer_kwargs)
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd, **kwargs)
        if name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, **kwargs)
        if name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, **kwargs)
        if name == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=wd, **kwargs)
        raise ValueError(f"Unsupported optimizer_name='{self.config.optimizer_name}'")

    def _build_scheduler(self):
        if self.config.scheduler_factory is not None:
            return self.config.scheduler_factory(self.optimizer)

        name = (self.config.scheduler_name or "none").lower()
        kwargs = dict(self.config.scheduler_kwargs)

        if name in {"none", "", "null"}:
            return None
        if name == "reduce_on_plateau":
            default = dict(mode=self._infer_monitor_mode(), factor=0.6, patience=10, min_lr=1e-6)
            default.update(kwargs)
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **default)
        if name == "onecycle":
            if "max_lr" not in kwargs:
                kwargs["max_lr"] = self.config.learning_rate
            if "steps_per_epoch" not in kwargs:
                kwargs["steps_per_epoch"] = max(1, len(self.train_loader))
            if "epochs" not in kwargs:
                kwargs["epochs"] = self.config.num_epochs
            return optim.lr_scheduler.OneCycleLR(self.optimizer, **kwargs)
        if name == "cosine":
            if "T_max" not in kwargs:
                kwargs["T_max"] = self.config.num_epochs
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **kwargs)
        raise ValueError(f"Unsupported scheduler_name='{self.config.scheduler_name}'")

    def _build_criterion(self):
        if self.config.criterion_fn is not None:
            return self.config.criterion_fn

        task = self.config.task_type.lower()
        loss_name = (self.config.loss_name or "").lower().strip()
        kwargs = dict(self.config.loss_kwargs)

        if task == "regression":
            if loss_name in {"", "mse", "mseloss"}:
                return nn.MSELoss(**kwargs)
            if loss_name in {"l1", "mae", "l1loss"}:
                return nn.L1Loss(**kwargs)
            if loss_name in {"smoothl1", "smooth_l1", "huber"}:
                return nn.SmoothL1Loss(**kwargs)
            raise ValueError(f"Unsupported regression loss_name='{self.config.loss_name}'")

        if task == "binary":
            if loss_name in {"", "bce", "bcewithlogits", "bcewithlogitsloss"}:
                return nn.BCEWithLogitsLoss(**kwargs)
            raise ValueError(f"Unsupported binary loss_name='{self.config.loss_name}'")

        if task == "multiclass":
            if loss_name in {"", "ce", "crossentropy", "crossentropyloss"}:
                return nn.CrossEntropyLoss(**kwargs)
            raise ValueError(f"Unsupported multiclass loss_name='{self.config.loss_name}'")

        if task == "multilabel":
            if loss_name in {"", "bce", "bcewithlogits", "bcewithlogitsloss"}:
                return nn.BCEWithLogitsLoss(**kwargs)
            raise ValueError(f"Unsupported multilabel loss_name='{self.config.loss_name}'")

        raise ValueError(f"Unsupported task_type='{self.config.task_type}'")

    # --------------------------- batch parsing --------------------------- #

    def _find_first_key(self, d: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
        for key in candidates:
            if key in d:
                return key
        for k in d.keys():
            kl = str(k).lower()
            if kl in {c.lower() for c in candidates}:
                return k
        return None

    def _split_batch(self, batch: Batch) -> Tuple[List[Any], Dict[str, Any], Any, Any]:
        if isinstance(batch, dict):
            target_key = self.config.batch_target_key or self._find_first_key(
                batch, ["y", "target", "targets", "label", "labels"]
            )
            if target_key is None:
                raise KeyError("Could not infer target key from dict batch.")

            meta_key = self.config.batch_meta_key or self._find_first_key(batch, ["meta", "metadata", "ids", "group_ids"])
            exclude = {target_key}
            if meta_key is not None:
                exclude.add(meta_key)

            if self.config.batch_input_keys:
                args = [batch[k] for k in self.config.batch_input_keys]
            else:
                primary_input_key = self._find_first_key(batch, ["x", "input", "inputs", "features", "data"])
                if primary_input_key is not None and primary_input_key not in exclude:
                    args = [batch[primary_input_key]]
                else:
                    args = [v for k, v in batch.items() if k not in exclude]

            kwargs: Dict[str, Any] = {}
            target = batch[target_key]
            meta = batch.get(meta_key) if meta_key is not None else None
            return args, kwargs, target, meta

        if isinstance(batch, (tuple, list)):
            if len(batch) < 2:
                raise ValueError("Tuple/list batch must contain at least inputs and targets.")
            args = list(batch[:-1])
            target = batch[-1]
            return args, {}, target, None

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _move_to_device(self, x: Any) -> Any:
        if torch.is_tensor(x):
            return x.to(self.device, non_blocking=self.config.non_blocking)
        if isinstance(x, dict):
            return {k: self._move_to_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            moved = [self._move_to_device(v) for v in x]
            return type(x)(moved) if isinstance(x, tuple) else moved
        return x

    def _extract_prediction_tensor(self, output: Any) -> torch.Tensor:
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            preferred = []
            if self.config.output_key is not None:
                preferred.append(self.config.output_key)
            preferred.extend(["pred", "preds", "prediction", "predictions", "logits", "output", "outputs", "yhat"])
            for key in preferred:
                if key in output and torch.is_tensor(output[key]):
                    return output[key]
            for value in output.values():
                if torch.is_tensor(value):
                    return value

        if isinstance(output, (tuple, list)):
            idx = self.config.output_index
            if 0 <= idx < len(output) and torch.is_tensor(output[idx]):
                return output[idx]
            for item in output:
                if torch.is_tensor(item):
                    return item

        raise TypeError("Could not extract prediction tensor from model output.")

    def _prepare_target(self, target: Any, pred: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)
        target = target.to(self.device, non_blocking=self.config.non_blocking)

        task = self.config.task_type.lower()
        if task == "regression":
            target = target.float()
            if pred.ndim == 1:
                return target.view(-1)
            if pred.ndim == 2 and pred.shape[-1] == 1 and target.ndim == 1:
                return target.view(-1, 1)
            return target.float().reshape_as(pred) if target.numel() == pred.numel() else target.float()

        if task == "binary":
            target = target.float()
            if pred.ndim == 2 and pred.shape[-1] == 1 and target.ndim == 1:
                target = target.unsqueeze(-1)
            return target.reshape_as(pred) if target.numel() == pred.numel() else target

        if task == "multiclass":
            return target.long().view(-1)

        if task == "multilabel":
            target = target.float()
            return target.reshape_as(pred) if target.numel() == pred.numel() else target

        raise ValueError(f"Unsupported task_type='{self.config.task_type}'")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.task_type.lower() == "multiclass":
            if pred.ndim == 1:
                raise ValueError("For multiclass classification, prediction tensor must have shape [B, C].")
            return self.criterion(pred, target)
        return self.criterion(pred, target)

    def _sigmoid_np(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-x))

    def _softmax_np(self, x: Array, axis: int = 1) -> Array:
        x = np.asarray(x, dtype=np.float64)
        z = x - x.max(axis=axis, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=axis, keepdims=True)

    def _prediction_array_for_metrics(self, y_pred_raw: Array) -> Array:
        task = self.config.task_type.lower()
        if task == "regression":
            return np.asarray(y_pred_raw)
        if task in {"binary", "multilabel"}:
            return self._sigmoid_np(y_pred_raw)
        if task == "multiclass":
            return self._softmax_np(y_pred_raw, axis=1)
        return np.asarray(y_pred_raw)

    def _target_array_for_metrics(self, y_true_raw: Array) -> Array:
        task = self.config.task_type.lower()
        if task == "multiclass":
            return np.asarray(y_true_raw).reshape(-1).astype(int)
        if task == "binary":
            return np.asarray(y_true_raw).reshape(-1).astype(int)
        if task == "multilabel":
            return np.asarray(y_true_raw).astype(int)
        return np.asarray(y_true_raw)

    def _infer_metric_names(self) -> List[str]:
        task = self.config.task_type.lower()
        if task == "regression":
            return ["loss", "rmse", "mae", "r2"]
        if task == "binary":
            return ["loss", "accuracy", "precision", "recall", "f1", "auroc"]
        if task == "multiclass":
            return ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auroc_ovr"]
        if task == "multilabel":
            return ["loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "auroc_macro"]
        return ["loss"]

    def _infer_monitor_mode(self) -> str:
        if self.config.monitor_mode in {"min", "max"}:
            return self.config.monitor_mode

        metric = self.config.monitor_metric.lower()
        lower_is_better = {
            "loss", "mse", "rmse", "mae", "nasa_score", "mape", "smape"
        }
        higher_is_better = {
            "r2", "accuracy", "precision", "recall", "f1", "auroc",
            "precision_macro", "recall_macro", "f1_macro", "auroc_macro", "auroc_ovr"
        }
        if metric in lower_is_better:
            return "min"
        if metric in higher_is_better:
            return "max"
        return "min"

    def _is_improvement(self, current: float) -> bool:
        if self.best_score is None:
            return True
        if self._infer_monitor_mode() == "min":
            return current < self.best_score
        return current > self.best_score

    @staticmethod
    def _jsonable(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, dict):
            return {k: Trainer._jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [Trainer._jsonable(v) for v in x]
        if hasattr(x, "__dict__") and not callable(x):
            try:
                return {k: Trainer._jsonable(v) for k, v in x.__dict__.items()}
            except Exception:
                return str(x)
        if callable(x):
            return getattr(x, "__name__", str(x))
        return x

    def _paths_for(self, tag: str) -> Tuple[str, str, str]:
        weights_path = os.path.join(self.config.save_path, f"{self.config.model_name}_{tag}.pth")
        ckpt_path = os.path.join(self.config.save_path, f"{self.config.model_name}_{tag}.ckpt")
        meta_path = os.path.join(self.config.save_path, f"{self.config.model_name}_{tag}_meta.json")
        return weights_path, ckpt_path, meta_path

    # --------------------------- core pass --------------------------- #

    def _forward_model(self, model_args: List[Any], model_kwargs: Dict[str, Any]) -> torch.Tensor:
        model_args = [self._move_to_device(arg) for arg in model_args]
        model_kwargs = {k: self._move_to_device(v) for k, v in model_kwargs.items()}
        output = self.model(*model_args, **model_kwargs)
        return self._extract_prediction_tensor(output)

    def _run_loader(self, loader: DataLoader, train: bool, epoch: Optional[int] = None) -> Dict[str, Any]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        num_batches = max(1, len(loader))
        collected_pred: List[Array] = []
        collected_true: List[Array] = []
        collected_meta: List[Any] = []

        desc = f"Epoch {epoch}/{self.config.num_epochs} [{'Train' if train else 'Val'}]" if epoch is not None else ("Training" if train else "Evaluating")
        iterator = tqdm(loader, desc=desc, leave=False)

        self.optimizer.zero_grad(set_to_none=True)

        grad_context = contextlib.nullcontext() if train else (torch.inference_mode() if self.config.use_inference_mode else torch.no_grad())
        with grad_context:
            for step_idx, batch in enumerate(iterator, start=1):
                model_args, model_kwargs, target, meta = self._split_batch(batch)

                with self._autocast_context():
                    pred = self._forward_model(model_args, model_kwargs)
                    target_t = self._prepare_target(target, pred)
                    loss = self._compute_loss(pred, target_t)

                if train:
                    scaled_loss = loss / max(1, self.config.accumulation_steps)
                    if self.scaler is not None and self._amp_enabled():
                        self.scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    if step_idx % max(1, self.config.accumulation_steps) == 0 or step_idx == len(loader):
                        if self.config.gradient_clip_value and self.config.gradient_clip_value > 0:
                            if self.scaler is not None and self._amp_enabled():
                                self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

                        if self.scaler is not None and self._amp_enabled():
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                        if self.scheduler is not None and self.config.scheduler_name and self.config.scheduler_name.lower() == "onecycle":
                            self.scheduler.step()

                total_loss += float(loss.detach().item())
                collected_pred.append(pred.detach().float().cpu().numpy())
                collected_true.append(target_t.detach().float().cpu().numpy())
                if meta is not None:
                    collected_meta.append(meta)

                iterator.set_postfix({"loss": f"{loss.detach().item():.4f}"})

        iterator.close()

        y_pred_raw = np.concatenate([np.atleast_1d(x) for x in collected_pred], axis=0) if collected_pred else np.array([])
        y_true_raw = np.concatenate([np.atleast_1d(x) for x in collected_true], axis=0) if collected_true else np.array([])

        y_pred_metric = self._prediction_array_for_metrics(y_pred_raw)
        y_true_metric = self._target_array_for_metrics(y_true_raw)
        meta_items = collected_meta if collected_meta else None

        if self.config.metric_transform is not None:
            y_true_metric, y_pred_metric = self.config.metric_transform(y_true_metric, y_pred_metric, meta_items)

        metrics = self._compute_metrics(
            y_true=y_true_metric,
            y_pred=y_pred_metric,
            mean_loss=float(total_loss / num_batches),
        )

        return {
            "metrics": metrics,
            "y_true": y_true_metric,
            "y_pred": y_pred_metric,
            "y_true_raw": y_true_raw,
            "y_pred_raw": y_pred_raw,
        }

    def _compute_metrics(self, y_true: Array, y_pred: Array, mean_loss: float) -> Dict[str, float]:
        task = self.config.task_type.lower()
        metrics: Dict[str, float] = {"loss": float(mean_loss)}

        if task == "regression":
            reg = RegressionMetrics.calculate(y_true, y_pred)
            metrics.update({k: v for k, v in reg.items() if k in {"rmse", "mae", "r2"}})
        elif task == "binary":
            metrics.update(ClassificationMetrics.calculate_binary(y_true, y_pred, self.config.multilabel_threshold))
        elif task == "multiclass":
            metrics.update(ClassificationMetrics.calculate_multiclass(y_true, y_pred))
        elif task == "multilabel":
            metrics.update(ClassificationMetrics.calculate_multilabel(y_true, y_pred, self.config.multilabel_threshold))

        for name, fn in self.config.custom_metrics.items():
            try:
                metrics[name] = float(fn(y_true, y_pred))
            except TypeError:
                metrics[name] = float(fn(y_true=y_true, y_pred=y_pred))

        return metrics

    # --------------------------- saving / loading --------------------------- #

    def save_history(self) -> None:
        path = os.path.join(self.config.save_path, f"{self.config.model_name}_history.json")
        serial = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        with open(path, "w") as f:
            json.dump(serial, f, indent=2)

    def save_checkpoint(self, epoch: int, tag: str) -> None:
        weights_path, ckpt_path, meta_path = self._paths_for(tag)
        torch.save(self.model.state_dict(), weights_path)

        preprocess = self._jsonable(self.preprocess) if self.preprocess is not None else None
        ckpt = {
            "epoch": int(epoch),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None and hasattr(self.scaler, "state_dict") else None,
            "best_score": float(self.best_score) if self.best_score is not None else None,
            "best_epoch": int(self.best_epoch),
            "history": self.history,
            "config": self._jsonable(self.config.__dict__),
            "preprocess": preprocess,
        }
        torch.save(ckpt, ckpt_path)

        meta = {
            "epoch": int(epoch),
            "best_epoch": int(self.best_epoch),
            "best_score": float(self.best_score) if self.best_score is not None else None,
            "monitor_metric": self.config.monitor_metric,
            "monitor_mode": self._infer_monitor_mode(),
            "weights_file": os.path.basename(weights_path),
            "ckpt_file": os.path.basename(ckpt_path),
            "task_type": self.config.task_type,
            "dataset_name": self.config.dataset_name,
        }
        if isinstance(preprocess, dict):
            meta.update(
                {
                    "n_features": preprocess.get("n_features") or len(preprocess.get("feature_cols", [])),
                    "sequence_length": preprocess.get("sequence_length", preprocess.get("seq_len")),
                    "label_mode": preprocess.get("label_mode"),
                }
            )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_checkpoint(self, path: str) -> None:
        obj = torch.load(path, map_location=self.device)
        if isinstance(obj, dict) and "model_state_dict" in obj:
            self.model.load_state_dict(obj["model_state_dict"])
            if obj.get("optimizer_state_dict") is not None:
                self.optimizer.load_state_dict(obj["optimizer_state_dict"])
            if self.scheduler is not None and obj.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(obj["scheduler_state_dict"])
            if self.scaler is not None and obj.get("scaler_state_dict") is not None and hasattr(self.scaler, "load_state_dict"):
                self.scaler.load_state_dict(obj["scaler_state_dict"])
            self.best_score = obj.get("best_score", self.best_score)
            self.best_epoch = obj.get("best_epoch", self.best_epoch)
            if obj.get("history") is not None:
                self.history = obj["history"]
            self.preprocess = obj.get("preprocess", self.preprocess)
        else:
            self.model.load_state_dict(obj)

    # --------------------------- reporting --------------------------- #

    def _print_epoch_summary(self, epoch: int, train_loss: float, val_metrics: Dict[str, float]) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        tqdm.write(f"\n{'=' * 80}")
        tqdm.write(f"Epoch [{epoch}/{self.config.num_epochs}] Summary")
        tqdm.write(f"{'=' * 80}")
        tqdm.write(f"  Train Loss:     {train_loss:.6f}")
        for name in self._all_val_metric_names:
            if name in val_metrics:
                tqdm.write(f"  {('Val ' + name).ljust(16)}: {val_metrics[name]:.6f}")
        tqdm.write(f"  LR:             {lr:.8f}")
        tqdm.write(f"  Monitor:        {self.config.monitor_metric} ({self._infer_monitor_mode()})")
        tqdm.write(f"{'=' * 80}\n")

    # --------------------------- public API --------------------------- #

    def train_epoch(self, epoch: int) -> float:
        return float(self._run_loader(self.train_loader, train=True, epoch=epoch)["metrics"]["loss"])

    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        return self._run_loader(self.val_loader, train=False, epoch=epoch)["metrics"]

    def train(self) -> Dict[str, List[float]]:
        print(f"\n{'=' * 80}")
        print(f"Training Model: {self.config.model_name}")
        print(f"{'=' * 80}")
        print("Configuration:")
        print(f"  Task:          {self.config.task_type}")
        print(f"  Epochs:        {self.config.num_epochs}")
        print(f"  LR:            {self.config.learning_rate}")
        print(f"  Batch Size:    {self.config.batch_size}")
        print(f"  Optimizer:     {self.config.optimizer_name}")
        print(f"  Scheduler:     {self.config.scheduler_name}")
        print(f"  AMP:           {self._amp_enabled()}")
        print(f"  Device:        {self.device}")
        print(f"  Save Path:     {self.config.save_path}")
        print(f"  Monitor:       {self.config.monitor_metric} ({self._infer_monitor_mode()})")
        print(f"{'=' * 80}\n")

        overall = tqdm(range(1, self.config.num_epochs + 1), desc="Overall Progress", position=0)
        last_epoch = 0

        for epoch in overall:
            last_epoch = epoch
            train_result = self._run_loader(self.train_loader, train=True, epoch=epoch)
            val_result = self._run_loader(self.val_loader, train=False, epoch=epoch)

            train_metrics = train_result["metrics"]
            val_metrics = val_result["metrics"]
            train_loss = float(train_metrics["loss"])

            if self.scheduler is not None and self.config.scheduler_name:
                sched_name = self.config.scheduler_name.lower()
                if sched_name == "reduce_on_plateau":
                    metric_for_sched = float(val_metrics.get(self.config.scheduler_metric, val_metrics["loss"]))
                    self.scheduler.step(metric_for_sched)
                elif sched_name == "cosine":
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            if self.config.compute_train_metrics:
                for m in self._base_metric_names:
                    if m in train_metrics:
                        self.history[f"train_{m}"].append(float(train_metrics[m]))
            for m in self._all_val_metric_names:
                if m in val_metrics:
                    self.history[f"val_{m}"].append(float(val_metrics[m]))
            self.history["learning_rate"].append(float(lr))

            postfix = {"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_metrics['loss']:.4f}", "lr": f"{lr:.2e}"}
            extra = self.config.progress_bar_extra_metric
            if extra and extra in val_metrics:
                postfix[extra] = f"{val_metrics[extra]:.4f}"
            overall.set_postfix(**postfix)

            if epoch % self.config.print_every_n_epochs == 0 or epoch == 1:
                self._print_epoch_summary(epoch, train_loss, val_metrics)

            monitor_key = self.config.monitor_metric
            if monitor_key not in val_metrics:
                raise KeyError(f"monitor_metric='{monitor_key}' not found in val_metrics keys={list(val_metrics.keys())}")
            monitor_value = float(val_metrics[monitor_key])

            if self._is_improvement(monitor_value):
                self.best_score = monitor_value
                self.best_epoch = epoch
                self.save_checkpoint(epoch, tag="best")

            if self.early_stopping.step(monitor_value, epoch):
                tqdm.write(f"\n⚠ Early stopping at epoch {epoch}")
                break

            if epoch % self.config.save_every_n_epochs == 0:
                self.save_history()
                self.save_checkpoint(epoch, tag="last")

        overall.close()
        self.save_checkpoint(last_epoch, tag="last")
        self.save_history()

        best_weights, _, _ = self._paths_for("best")
        if os.path.exists(best_weights):
            self.model.load_state_dict(torch.load(best_weights, map_location=self.device))

        print(f"\n{'=' * 80}")
        print("Training Completed!")
        print(f"{'=' * 80}")
        print(f"  Best Epoch:      {self.best_epoch}")
        print(f"  Best({self.config.monitor_metric}): {self.best_score}")
        print(f"  Saved to:        {self.config.save_path}")
        print(f"{'=' * 80}\n")

        return self.history

    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict[str, float], Array, Array]:
        result = self._run_loader(test_loader, train=False, epoch=None)
        return result["metrics"], result["y_pred"], result["y_true"]
