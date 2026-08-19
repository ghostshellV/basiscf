from __future__ import annotations

"""
Generic CMAPSS training driver built on top of the rewritten generic Trainer.

Design goals
------------
1. Keep the script easy to use for CMAPSS right now.
2. Keep the training loop generic enough for future time-series datasets/models.
3. Avoid baking CMAPSS-specific logic into the Trainer itself.
4. Evaluate validation/test metrics on denormalised RUL via metric_transform.
5. Save per-run config, preprocess metadata, histories, and test metrics.
"""

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Project path bootstrap
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer the project modules. If the user is testing this file standalone,
# allow fallback to the rewritten uploaded files.
try:
    from src.data_loader.cmapss.v2.CMAPSSTorchDataloader_v2 import CMAPSSTorchDataset, CMAPSSPreprocessInfo
except Exception:  # pragma: no cover - fallback for standalone testing
    from CMAPSSTorchDataset_rewritten import CMAPSSTorchDataset, CMAPSSPreprocessInfo  # type: ignore

try:
    from src.trainer.Trainer_v2 import (
        NASAScore,
        Trainer,
        TrainingConfig,
        make_cmapss_metric_transform,
    )
except Exception:  # pragma: no cover - fallback for standalone testing
    from Trainer_rewritten import NASAScore, Trainer, TrainingConfig, make_cmapss_metric_transform  # type: ignore


# -----------------------------------------------------------------------------
# Default paths
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "processed" / "CMAPSS" / "data")
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "outputs" / "cmapss" / "v2")


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    epochs: int = 260
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    optimizer_name: str = "adamw"
    scheduler_name: Optional[str] = "reduce_on_plateau"
    scheduler_kwargs: Dict[str, Any] = field(default_factory=lambda: {"factor": 0.6, "patience": 12, "min_lr": 1e-6})
    monitor_metric: str = "loss"
    early_stopping_patience: int = 25
    early_stopping_start_epoch: int = 80
    gradient_clip_value: float = 1.0
    use_amp: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    loss_name: Optional[str] = None
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)


DEFAULT_MODEL_SPECS: Dict[str, ModelSpec] = {
    "lstm": ModelSpec(
        name="lstm",
        epochs=260,
        learning_rate=1e-3,
        model_kwargs={"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
    ),
    "gru": ModelSpec(
        name="gru",
        epochs=260,
        learning_rate=1e-3,
        model_kwargs={"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
    ),
    "cnn_lstm": ModelSpec(
        name="cnn_lstm",
        epochs=260,
        learning_rate=1e-3,
        model_kwargs={"hidden_size": 128, "num_layers": 2, "dropout": 0.2},
    ),
    "transformer": ModelSpec(
        name="transformer",
        epochs=260,
        learning_rate=5e-4,
        model_kwargs={"d_model": 128, "nhead": 4, "num_layers": 2, "dropout": 0.2},
    ),
    "star": ModelSpec(
        name="star",
        epochs=260,
        learning_rate=5e-4,
        model_kwargs={
            "patch_len": 5,
            "num_scales": 3,
            "d_model": 128,
            "nhead": 4,
            "ff_dim": 256,
            "dropout": 0.2,
        },
    ),
}


def create_model(
    model_name: str,
    input_size: int,
    device: torch.device,
    seq_len: int,
    max_rul: float,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """
    Factory for sequence models.

    The contract is intentionally simple:
      - input_size: number of input features presented to the main encoder
      - seq_len: sequence length
      - max_rul: useful for some CMAPSS-specific heads (e.g. STAR)
      - model_kwargs: extra architecture parameters

    Add new time-series models here without touching the Trainer.
    """
    kwargs = dict(model_kwargs or {})
    model_name = model_name.lower()

    if model_name == "lstm":
        from src.models.cmapss.LSTMModel import LSTMModel

        model = LSTMModel(input_size=input_size, **kwargs)
    elif model_name == "gru":
        from src.models.cmapss.GRUModel import GRUModel

        model = GRUModel(input_size=input_size, **kwargs)
    elif model_name == "cnn_lstm":
        from src.models.cmapss.CNNLSTMModel import CNNLSTMModel

        model = CNNLSTMModel(input_size=input_size, **kwargs)
    elif model_name == "transformer":
        from src.models.cmapss.TransformerModel import TransformerModel

        model = TransformerModel(input_size=input_size, **kwargs)
    elif model_name == "star":
        from src.models.cmapss.STARModel import STARModel

        model = STARModel(input_dim=input_size, seq_len=seq_len, max_rul=max_rul, **kwargs)
    else:
        raise ValueError(f"Unknown model_name='{model_name}'")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name.upper()} parameters: {n_params:,}")
    return model


# -----------------------------------------------------------------------------
# Preprocess metadata
# -----------------------------------------------------------------------------


def info_to_preprocess_dict(info: CMAPSSPreprocessInfo) -> Dict[str, Any]:
    feature_cols = [c for c in info.feature_list if c not in {"unit_number", "time_cycles", "RUL"}]
    return {
        "dataset_name": "cmapss",
        "subset": info.subset,
        "feature_list": info.feature_list,
        "feature_cols": feature_cols,
        "n_features": info.n_features,
        "n_context": info.n_context,
        "seq_len": info.seq_len,
        "batch_size": info.batch_size,
        "smoothing_window": info.smoothing_window,
        "use_clustering": info.use_clustering,
        "split_context": info.split_context,
        "feature_mode": getattr(info, "feature_mode", "selected"),
        "piecewise_rul_cap": getattr(info, "piecewise_rul_cap", None),
        "val_strategy": getattr(info, "val_strategy", None),
        "test_last_window_only": getattr(info, "test_last_window_only", None),
        "train_unit_ids": getattr(info, "train_unit_ids", None),
        "val_unit_ids": getattr(info, "val_unit_ids", None),
        "global_min": np.asarray(info.global_min, dtype=np.float32).tolist(),
        "global_range": np.asarray(info.global_range, dtype=np.float32).tolist(),
        "rul_min": float(info.rul_min),
        "rul_range": float(info.rul_range),
        "max_rul": float(info.rul_range),
        "label_mode": "piecewise_rul" if getattr(info, "piecewise_rul_cap", None) is not None else "raw_rul",
        "target_normalisation": True,
    }


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    data_dir: str = DEFAULT_DATA_DIR
    subsets: List[str] = field(default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"])
    seq_len: int = 50
    batch_size: Optional[int] = 32
    smoothing_window: int = 3
    split_context: bool = False
    seed: int = 42
    feature_mode: str = "selected"
    piecewise_rul_cap: Optional[int] = 125
    val_strategy: str = "engine"
    val_fraction: float = 0.20
    val_tail_multiple: int = 2
    test_last_window_only: bool = True
    shuffle_train: bool = True
    target_normalisation: bool = True
    cluster_by_operating_condition: bool = True
    n_clusters: int = 6
    num_workers: int = 0
    pin_memory: bool = False
    drop_constant_features: bool = False


@dataclass
class RunConfig:
    output_root: str = DEFAULT_OUTPUT_ROOT
    device: Optional[str] = None
    seed: int = 42
    save_test_predictions: bool = True
    compute_train_metrics: bool = False
    progress_bar_extra_metric: str = "nasa_score"


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


def build_training_config(
    *,
    spec: ModelSpec,
    subset: str,
    save_path: str,
    batch_size: int,
    preprocess: Dict[str, Any],
    compute_train_metrics: bool,
    progress_bar_extra_metric: Optional[str],
) -> TrainingConfig:
    return TrainingConfig(
        task_type="regression",
        num_epochs=spec.epochs,
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
        batch_size=batch_size,
        gradient_clip_value=spec.gradient_clip_value,
        optimizer_name=spec.optimizer_name,
        scheduler_name=spec.scheduler_name,
        scheduler_kwargs=dict(spec.scheduler_kwargs),
        loss_name=spec.loss_name,
        loss_kwargs=dict(spec.loss_kwargs),
        use_amp=spec.use_amp,
        dataset_name="cmapss",
        save_path=save_path,
        model_name=f"{spec.name}_{subset}",
        early_stopping_patience=spec.early_stopping_patience,
        early_stopping_start_epoch=spec.early_stopping_start_epoch,
        monitor_metric=spec.monitor_metric,
        compute_train_metrics=compute_train_metrics,
        custom_metrics={"nasa_score": NASAScore.compute},
        progress_bar_extra_metric=progress_bar_extra_metric,
        metric_transform=make_cmapss_metric_transform(preprocess),
    )



def evaluate_and_save(
    *,
    trainer: Trainer,
    test_loader: DataLoader,
    run_dir: str,
    save_predictions: bool,
) -> Dict[str, Any]:
    metrics, y_pred, y_true = trainer.evaluate(test_loader)

    metrics_path = os.path.join(run_dir, "test_metrics.json")
    _safe_json_dump(metrics_path, {k: float(v) for k, v in metrics.items()})

    if save_predictions:
        pred_payload = {
            "y_true": np.asarray(y_true).reshape(-1).tolist(),
            "y_pred": np.asarray(y_pred).reshape(-1).tolist(),
        }
        _safe_json_dump(os.path.join(run_dir, "test_predictions.json"), pred_payload)

    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}



def train_single_model(
    *,
    model_spec: ModelSpec,
    subset: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    preprocess: Dict[str, Any],
    device: torch.device,
    output_root: str,
    compute_train_metrics: bool,
    progress_bar_extra_metric: Optional[str],
    save_test_predictions: bool,
) -> Dict[str, Any]:
    model_name = model_spec.name
    input_size = int(preprocess["n_features"])
    seq_len = int(preprocess["seq_len"])
    max_rul = float(preprocess.get("max_rul", preprocess.get("rul_range", 1.0)))

    run_dir = os.path.join(output_root, subset, model_name)
    os.makedirs(run_dir, exist_ok=True)

    model = create_model(
        model_name=model_name,
        input_size=input_size,
        device=device,
        seq_len=seq_len,
        max_rul=max_rul,
        model_kwargs=model_spec.model_kwargs,
    )

    train_cfg = build_training_config(
        spec=model_spec,
        subset=subset,
        save_path=run_dir,
        batch_size=int(train_loader.batch_size or preprocess["batch_size"]),
        preprocess=preprocess,
        compute_train_metrics=compute_train_metrics,
        progress_bar_extra_metric=progress_bar_extra_metric,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=train_cfg,
        preprocess=preprocess,
    )

    _safe_json_dump(os.path.join(run_dir, "preprocess.json"), preprocess)
    _safe_json_dump(os.path.join(run_dir, "training_config.json"), asdict(train_cfg))
    _safe_json_dump(os.path.join(run_dir, "model_spec.json"), asdict(model_spec))

    history = trainer.train()
    test_result = evaluate_and_save(
        trainer=trainer,
        test_loader=test_loader,
        run_dir=run_dir,
        save_predictions=save_test_predictions,
    )

    print(f"[TEST] {subset} | {model_name}: {json.dumps(test_result['metrics'], indent=2, default=_json_default)}")

    return {
        "history": history,
        "test_metrics": test_result["metrics"],
        "run_dir": run_dir,
    }


# -----------------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------------


# def load_cmapss_subset(data_cfg: DataConfig, subset: str) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
#     ds = CMAPSSTorchDataset(data_dir=data_cfg.data_dir)
#     train_loader, val_loader, test_loader, info = ds.load(
#         subset=subset,
#         seq_len=data_cfg.seq_len,
#         batch_size=data_cfg.batch_size,
#         smoothing_window=data_cfg.smoothing_window,
#         split_context=data_cfg.split_context,
#         seed=data_cfg.seed,
#         feature_mode=data_cfg.feature_mode,
#         piecewise_rul_cap=data_cfg.piecewise_rul_cap,
#         val_strategy=data_cfg.val_strategy,
#         val_fraction=data_cfg.val_fraction,
#         val_tail_multiple=data_cfg.val_tail_multiple,
#         test_last_window_only=data_cfg.test_last_window_only,
#         shuffle_train=data_cfg.shuffle_train,
#         target_normalisation=data_cfg.target_normalisation,
#         cluster_by_operating_condition=data_cfg.cluster_by_operating_condition,
#         n_clusters=data_cfg.n_clusters,
#         num_workers=data_cfg.num_workers,
#         pin_memory=data_cfg.pin_memory,
#         drop_constant_features=data_cfg.drop_constant_features,
#     )
#     preprocess = info_to_preprocess_dict(info)
#     return train_loader, val_loader, test_loader, preprocess

def load_cmapss_subset(
    data_cfg: DataConfig,
    subset: str,
    return_dataset: bool = False,
):
    ds = CMAPSSTorchDataset(data_dir=data_cfg.data_dir)
    train_loader, val_loader, test_loader, info = ds.load(
        subset=subset,
        seq_len=data_cfg.seq_len,
        batch_size=data_cfg.batch_size,
        smoothing_window=data_cfg.smoothing_window,
        split_context=data_cfg.split_context,
        seed=data_cfg.seed,
        feature_mode=data_cfg.feature_mode,
        piecewise_rul_cap=data_cfg.piecewise_rul_cap,
        val_strategy=data_cfg.val_strategy,
        val_fraction=data_cfg.val_fraction,
        val_tail_multiple=data_cfg.val_tail_multiple,
        test_last_window_only=data_cfg.test_last_window_only,
        shuffle_train=data_cfg.shuffle_train,
        target_normalisation=data_cfg.target_normalisation,
        cluster_by_operating_condition=data_cfg.cluster_by_operating_condition,
        n_clusters=data_cfg.n_clusters,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_constant_features=data_cfg.drop_constant_features,
    )
    preprocess = info_to_preprocess_dict(info)

    if return_dataset:
        return train_loader, val_loader, test_loader, preprocess, ds

    return train_loader, val_loader, test_loader, preprocess

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train time-series models on NASA CMAPSS using the generic Trainer.")

    # Data / run
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # CMAPSS dataset
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoothing-window", type=int, default=3)
    parser.add_argument("--split-context", action="store_true")
    parser.add_argument("--feature-mode", type=str, default="selected", choices=["selected", "full"])
    parser.add_argument("--piecewise-rul-cap", type=int, default=125)
    parser.add_argument("--val-strategy", type=str, default="engine", choices=["engine", "tail"])
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--val-tail-multiple", type=int, default=2)
    parser.add_argument("--no-test-last-window-only", action="store_true")
    parser.add_argument("--no-shuffle-train", action="store_true")
    parser.add_argument("--no-target-normalisation", action="store_true")
    parser.add_argument("--no-cluster-by-operating-condition", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--drop-constant-features", action="store_true")

    # Models / trainer
    parser.add_argument("--models", nargs="+", default=["lstm", "cnn_lstm", "transformer", "star"])
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all selected models.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override LR for all selected models.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay for all selected models.")
    parser.add_argument("--monitor-metric", type=str, default=None, help="Override monitor metric for all selected models.")
    parser.add_argument("--compute-train-metrics", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-save-test-predictions", action="store_true")

    return parser.parse_args()



def build_configs_from_args(args: argparse.Namespace) -> Tuple[DataConfig, RunConfig, List[ModelSpec]]:
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        subsets=list(args.subsets),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        smoothing_window=args.smoothing_window,
        split_context=bool(args.split_context),
        seed=args.seed,
        feature_mode=args.feature_mode,
        piecewise_rul_cap=args.piecewise_rul_cap,
        val_strategy=args.val_strategy,
        val_fraction=args.val_fraction,
        val_tail_multiple=args.val_tail_multiple,
        test_last_window_only=not args.no_test_last_window_only,
        shuffle_train=not args.no_shuffle_train,
        target_normalisation=not args.no_target_normalisation,
        cluster_by_operating_condition=not args.no_cluster_by_operating_condition,
        n_clusters=args.n_clusters,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_constant_features=bool(args.drop_constant_features),
    )

    run_cfg = RunConfig(
        output_root=args.output_root,
        device=args.device,
        seed=args.seed,
        save_test_predictions=not args.no_save_test_predictions,
        compute_train_metrics=bool(args.compute_train_metrics),
        progress_bar_extra_metric="nasa_score",
    )

    model_specs: List[ModelSpec] = []
    for model_name in args.models:
        if model_name not in DEFAULT_MODEL_SPECS:
            raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(DEFAULT_MODEL_SPECS)}")
        base = DEFAULT_MODEL_SPECS[model_name]
        spec = ModelSpec(**asdict(base))
        if args.epochs is not None:
            spec.epochs = args.epochs
        if args.learning_rate is not None:
            spec.learning_rate = args.learning_rate
        if args.weight_decay is not None:
            spec.weight_decay = args.weight_decay
        if args.monitor_metric is not None:
            spec.monitor_metric = args.monitor_metric
        if args.no_amp:
            spec.use_amp = False
        model_specs.append(spec)

    return data_cfg, run_cfg, model_specs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    data_cfg, run_cfg, model_specs = build_configs_from_args(args)

    seed_everything(run_cfg.seed)
    device = choose_device(run_cfg.device)

    print("=" * 90)
    print("CMAPSS generic training driver")
    print("=" * 90)
    print(f"Device:      {device}")
    print(f"Data dir:    {data_cfg.data_dir}")
    print(f"Output root: {run_cfg.output_root}")
    print(f"Subsets:     {data_cfg.subsets}")
    print(f"Models:      {[m.name for m in model_specs]}")
    print("=" * 90)

    os.makedirs(run_cfg.output_root, exist_ok=True)
    _safe_json_dump(os.path.join(run_cfg.output_root, "run_config.json"), {
        "data": asdict(data_cfg),
        "run": asdict(run_cfg),
        "models": [asdict(m) for m in model_specs],
    })

    all_results: Dict[str, Dict[str, Any]] = {}

    for subset in data_cfg.subsets:
        print(f"\n{'#' * 90}")
        print(f"# Loading subset: {subset}")
        print(f"{'#' * 90}")

        train_loader, val_loader, test_loader, preprocess = load_cmapss_subset(data_cfg, subset)

        subset_results: Dict[str, Any] = {}
        for spec in model_specs:
            print(f"\n{'=' * 90}")
            print(f"Training {spec.name.upper()} on {subset}")
            print(f"{'=' * 90}")
            result = train_single_model(
                model_spec=spec,
                subset=subset,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                preprocess=preprocess,
                device=device,
                output_root=run_cfg.output_root,
                compute_train_metrics=run_cfg.compute_train_metrics,
                progress_bar_extra_metric=run_cfg.progress_bar_extra_metric,
                save_test_predictions=run_cfg.save_test_predictions,
            )
            subset_results[spec.name] = {
                "run_dir": result["run_dir"],
                "test_metrics": result["test_metrics"],
            }

        all_results[subset] = subset_results
        _safe_json_dump(os.path.join(run_cfg.output_root, subset, "summary.json"), subset_results)

    _safe_json_dump(os.path.join(run_cfg.output_root, "all_results.json"), all_results)

    print(f"\n{'=' * 90}")
    print("Training complete.")
    print(f"Results saved under: {run_cfg.output_root}")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    main()
