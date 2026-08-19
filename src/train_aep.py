"""
train_aep.py — Training script for Appliance Energy Prediction (AEP).

Loads the energydata_complete.csv via AEPTorchDataset, trains one or more
sequence models (LSTM, GRU, CNN-LSTM, Transformer), and evaluates on the
held-out test split.

Models are saved under outputs/AEP/<model_name>/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Project path bootstrap
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader.aep.AEPTorchDataset import AEPTorchDataset

from src.trainer.Trainer_v2 import Trainer, TrainingConfig

from src.models.aep.LSTMModel import LSTMModel
from src.models.aep.GRUModel import GRUModel
from src.models.aep.CNNLSTMModel import CNNLSTMModel
from src.models.aep.TransformerModel import TransformerModel

# -----------------------------------------------------------------------------
# Default paths
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "processed" / "AEP" / "dataset")
DEFAULT_FILENAME = "energydata_complete.csv"
DEFAULT_OUTPUT_ROOT = str(PROJECT_ROOT / "outputs" / "AEP")


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    return torch.device(
        "cuda"
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        else "cpu"
    )


def _safe_json_dump(path: str, payload: Any) -> None:
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
    scheduler_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"factor": 0.6, "patience": 12, "min_lr": 1e-6}
    )
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
        model_kwargs={
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.2,
        },
    ),
}


def create_model(
    model_name: str,
    input_size: int,
    device: torch.device,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    """Instantiate a model by name and move it to *device*."""
    kwargs = dict(model_kwargs or {})
    name = model_name.lower()

    if name == "lstm":
        model = LSTMModel(input_size=input_size, **kwargs)
    elif name == "gru":
        model = GRUModel(input_size=input_size, **kwargs)
    elif name in {"cnn_lstm", "cnnlstm"}:
        model = CNNLSTMModel(input_size=input_size, **kwargs)
    elif name == "transformer":
        model = TransformerModel(input_size=input_size, **kwargs)
    else:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            f"Choose from: lstm, gru, cnn_lstm, transformer"
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {model_name.upper()} — {n_params:,} parameters")
    return model


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_data(
    data_dir: str = DEFAULT_DATA_DIR,
    filename: str = DEFAULT_FILENAME,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load the AEP dataset and return train/val/test DataLoaders plus a
    preprocess metadata dict for checkpoint embedding.
    """
    ds = AEPTorchDataset(data_dir=data_dir)
    train_loader, val_loader, test_loader = ds.load(filename)

    # Extract shape info from the first batch
    sample_x, sample_y = next(iter(train_loader))
    n_features = sample_x.shape[-1]
    seq_len = sample_x.shape[-2]
    batch_size = sample_x.shape[0]

    # Feature columns kept after the dataset drops the 6 low-value columns.
    # The dataset's X input contains sensor + weather features (excluding
    # weekday_sin/weekday_cos which are split into context Z).
    input_feature_cols = [
        "nsm", "lights", "T1", "T2", "T3",
        "T4", "RH_4", "T5", "T6",
        "T7", "RH_7", "T8", "RH_8",
        "T_out", "Press_mm_hg", "RH_out", "Windspeed",
        "Visibility", "Tdewpoint", "Appliances",
    ]
    # Trim to actual feature count
    input_feature_cols = input_feature_cols[:n_features]

    preprocess: Dict[str, Any] = {
        "dataset_name": "aep",
        "data_dir": data_dir,
        "filename": filename,
        "feature_cols": input_feature_cols,
        "n_features": int(n_features),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "n_past": 12,
        "n_future": 1,
        "normalisation": "minmax",
        "target_col": "Appliances",
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
    }

    print(f"\nLoaded AEP data from {data_dir}/{filename}")
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    print(f"  Input shape:  ({seq_len}, {n_features})")

    return train_loader, val_loader, test_loader, preprocess


# -----------------------------------------------------------------------------
# Training config builder
# -----------------------------------------------------------------------------


def build_training_config(
    *,
    spec: ModelSpec,
    save_path: str,
    batch_size: int,
    preprocess: Dict[str, Any],
    compute_train_metrics: bool = False,
    progress_bar_extra_metric: Optional[str] = None,
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
        dataset_name="aep",
        save_path=save_path,
        model_name=f"{spec.name}_best",
        early_stopping_patience=spec.early_stopping_patience,
        early_stopping_start_epoch=spec.early_stopping_start_epoch,
        monitor_metric=spec.monitor_metric,
        compute_train_metrics=compute_train_metrics,
        progress_bar_extra_metric=progress_bar_extra_metric,
    )


# -----------------------------------------------------------------------------
# Evaluate and save results
# -----------------------------------------------------------------------------


def evaluate_and_save(
    *,
    trainer: Trainer,
    test_loader: DataLoader,
    run_dir: str,
    save_predictions: bool = True,
) -> Dict[str, Any]:
    metrics, y_pred, y_true = trainer.evaluate(test_loader)

    _safe_json_dump(
        os.path.join(run_dir, "test_metrics.json"),
        {k: float(v) for k, v in metrics.items()},
    )

    if save_predictions:
        _safe_json_dump(
            os.path.join(run_dir, "test_predictions.json"),
            {
                "y_true": np.asarray(y_true).reshape(-1).tolist(),
                "y_pred": np.asarray(y_pred).reshape(-1).tolist(),
            },
        )

    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred}


# -----------------------------------------------------------------------------
# Single model training loop
# -----------------------------------------------------------------------------


def train_single_model(
    *,
    model_spec: ModelSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    preprocess: Dict[str, Any],
    device: torch.device,
    output_root: str,
    compute_train_metrics: bool = False,
    progress_bar_extra_metric: Optional[str] = None,
    save_test_predictions: bool = True,
) -> Dict[str, Any]:
    model_name = model_spec.name
    input_size = int(preprocess["n_features"])

    run_dir = os.path.join(output_root, model_name)
    os.makedirs(run_dir, exist_ok=True)

    model = create_model(
        model_name=model_name,
        input_size=input_size,
        device=device,
        model_kwargs=model_spec.model_kwargs,
    )

    train_cfg = build_training_config(
        spec=model_spec,
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

    # Persist configs
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

    print(f"[TEST] {model_name}: {json.dumps(test_result['metrics'], indent=2, default=_json_default)}")

    return {
        "history": history,
        "test_metrics": test_result["metrics"],
        "run_dir": run_dir,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


@dataclass
class DataConfig:
    data_dir: str = DEFAULT_DATA_DIR
    filename: str = DEFAULT_FILENAME


@dataclass
class RunConfig:
    output_root: str = DEFAULT_OUTPUT_ROOT
    device: Optional[str] = None
    seed: int = 42
    save_test_predictions: bool = True
    compute_train_metrics: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train time-series models on the AEP (Appliance Energy Prediction) dataset."
    )

    # Data
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME)

    # Run
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Models
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "gru", "cnn_lstm", "transformer"],
        help="Models to train. Available: lstm, gru, cnn_lstm, transformer",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for all selected models.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override LR for all.")
    parser.add_argument("--weight-decay", type=float, default=None, help="Override weight decay for all.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument("--no-save-test-predictions", action="store_true")
    parser.add_argument("--compute-train-metrics", action="store_true")

    return parser.parse_args()


def build_configs_from_args(args: argparse.Namespace) -> Tuple[DataConfig, RunConfig, List[ModelSpec]]:
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        filename=args.filename,
    )

    run_cfg = RunConfig(
        output_root=args.output_root,
        device=args.device,
        seed=args.seed,
        save_test_predictions=not args.no_save_test_predictions,
        compute_train_metrics=bool(args.compute_train_metrics),
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
    print(f"\nUsing device: {device}")

    # ── load data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, preprocess = load_data(
        data_dir=data_cfg.data_dir,
        filename=data_cfg.filename,
    )

    # ── persist preprocess config ────────────────────────────────────────────
    os.makedirs(run_cfg.output_root, exist_ok=True)
    _safe_json_dump(os.path.join(run_cfg.output_root, "preprocess.json"), preprocess)

    # ── summary banner ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("AEP — TRAINING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Features:        {preprocess['n_features']}")
    print(f"  Sequence length: {preprocess['seq_len']}")
    print(f"  Batch size:      {preprocess['batch_size']}")
    print(f"  Train samples:   {preprocess['n_train']:,}")
    print(f"  Val samples:     {preprocess['n_val']:,}")
    print(f"  Test samples:    {preprocess['n_test']:,}")
    print(f"  Train batches:   {len(train_loader)}")
    print(f"  Val batches:     {len(val_loader)}")
    print(f"  Test batches:    {len(test_loader)}")
    print(f"  Models:          {[s.name for s in model_specs]}")
    print(f"{'=' * 70}")

    # ── training loop ────────────────────────────────────────────────────────
    all_results: Dict[str, Any] = {}

    for spec in model_specs:
        print(f"\n{'#' * 80}")
        print(f"# Training {spec.name.upper()}")
        print(f"{'#' * 80}")

        result = train_single_model(
            model_spec=spec,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            preprocess=preprocess,
            device=device,
            output_root=run_cfg.output_root,
            compute_train_metrics=run_cfg.compute_train_metrics,
            save_test_predictions=run_cfg.save_test_predictions,
        )
        all_results[spec.name] = result

    # ── final summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ALL TRAINING COMPLETE")
    print(f"{'=' * 70}")
    for model_name, res in all_results.items():
        tm = res["test_metrics"]
        print(
            f"  {model_name.upper():15s}  "
            f"RMSE={tm.get('rmse', float('nan')):.4f}  "
            f"MAE={tm.get('mae', float('nan')):.4f}  "
            f"R2={tm.get('r2', float('nan')):.4f}"
        )
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()

# ======================================================================
# ALL TRAINING COMPLETE
# ======================================================================
#   LSTM             RMSE=0.0690  MAE=0.0315  R2=0.5702
#   GRU              RMSE=0.0696  MAE=0.0323  R2=0.5622
#   CNN_LSTM         RMSE=0.0851  MAE=0.0439  R2=0.3453
#   TRANSFORMER      RMSE=0.0716  MAE=0.0315  R2=0.5373
# ======================================================================