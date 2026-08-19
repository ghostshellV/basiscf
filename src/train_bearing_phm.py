from __future__ import annotations

import json
import os
import sys
import random
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any, List, Set

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── project root on sys.path ────────────────────────────────────────────────
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader.ieee_phm.IEEEPHMDataLoader import build_dataloaders
from src.data_loader.ieee_phm.DataConfig import DataConfig

from src.models.ieee_phm.LSTMModel import LSTMModel
from src.models.ieee_phm.GRUModel import GRUModel
from src.models.ieee_phm.CNNLSTMModel import CNNLSTMModel
from src.models.ieee_phm.TransformerModel import TransformerModel

from src.trainer.Trainer import Trainer, TrainingConfig

# ── paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = project_root / "data" / "processed" / "ieee_phm"
OUTPUT_DIR = project_root / "outputs" / "ieee_phm_bearing"
SAVE_MODEL_DIR = OUTPUT_DIR / "saved_models"


# ── reproducibility ─────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── model factory ───────────────────────────────────────────────────────────
def create_model(model_type: str, input_size: int, device: torch.device):
    if model_type == "lstm":
        model = LSTMModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "gru":
        model = GRUModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "cnn_lstm":
        model = CNNLSTMModel(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2)
    elif model_type == "transformer":
        model = TransformerModel(input_size=input_size, d_model=128, nhead=4, num_layers=2, dropout=0.2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)
    print(f"{model_type.upper()} params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ── data loading ─────────────────────────────────────────────────────────────
def load_preprocessed_data() -> Tuple[dict, dict, List[str], DataConfig, dict]:
    """Load the cleaned bearing data from the pickle saved during EDA/preprocessing."""
    pkl_path = PROCESSED_DIR / "cleaned_bearing_data.pkl"
    with open(pkl_path, "rb") as f:
        data_dict = pickle.load(f)

    cleaned_train = data_dict["train"]
    cleaned_test = data_dict["test"]
    feature_cols = data_dict["feature_cols"]
    config = data_dict["config"]
    metadata = data_dict["metadata"]
    return cleaned_train, cleaned_test, feature_cols, config, metadata


def load_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, Any, Dict, List[str]]:
    """Build train / val / test loaders and return all info needed for training."""
    cleaned_train, cleaned_test, feature_cols, config, _ = load_preprocessed_data()

    train_loader, val_loader, test_loader, normalizer, metadata = build_dataloaders(
        cleaned_train, cleaned_test, feature_cols, config
    )
    return train_loader, val_loader, test_loader, normalizer, metadata, feature_cols


def build_preprocess_dict(
    feature_cols: List[str],
    metadata: Dict[str, Any],
    normalizer: Any,
) -> Dict[str, Any]:
    """
    Assemble a `preprocess` dict that the generic Trainer can embed in checkpoints.
    This mirrors the structure used by train_cmapss.py so that downstream code
    (evaluation, counterfactual generation) can rely on a uniform schema.
    """
    # Normalizer params (already JSON-safe scalars / lists)
    normalizer_params = normalizer.get_params() if hasattr(normalizer, "get_params") else {}

    preprocess = {
        "dataset": "ieee_phm_bearing",
        "feature_cols": list(feature_cols),
        "sequence_length": int(metadata.get("seq_len", 30)),
        "n_features": int(metadata.get("n_features", len(feature_cols))),
        "scaler_type": metadata.get("scaler_type", "minmax"),
        "max_rul": float(normalizer_params.get("target_max", metadata.get("y_train_original_range", [0, 1])[-1])),
        "normalizer_params": normalizer_params,
        "n_train": int(metadata.get("n_train", 0)),
        "n_val": int(metadata.get("n_val", 0)),
        "n_test": int(metadata.get("n_test", 0)),
    }
    return preprocess


# ── training ─────────────────────────────────────────────────────────────────
def train_single_model(
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    preprocess: Dict[str, Any],
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    save_path: str,
):
    input_size = len(preprocess["feature_cols"])
    model = create_model(model_type, input_size, device)

    cfg = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=train_loader.batch_size,
        early_stopping_patience=25,
        early_stopping_start_epoch=90,
        gradient_clip_value=1.0,
        save_path=save_path,
        dataset_name="ieee_phm_bearing",
        model_name=f"{model_type}_best",
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=cfg,
        preprocess=preprocess,
    )

    history = trainer.train()
    return history, trainer


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    seed_everything(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7 else "cpu"
    )
    print(f"\nUsing device: {device}")

    # ── load data ────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, normalizer, metadata, feature_cols = load_dataloaders()

    # ── build preprocess dict (consumed by Trainer & saved in checkpoints) ───
    preprocess = build_preprocess_dict(feature_cols, metadata, normalizer)

    # ── save preprocess dict for later use ───────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    preprocess_path = OUTPUT_DIR / "preprocess.json"
    with open(preprocess_path, "w") as f:
        json.dump(preprocess, f, indent=2)
    print(f"Preprocess config saved to {preprocess_path}")

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  n_features:    {metadata['n_features']}")
    print(f"  seq_len:       {metadata['seq_len']}")
    print(f"  Train:         {metadata['n_train']} samples")
    print(f"  Validation:    {metadata['n_val']} samples")
    print(f"  Test:          {metadata['n_test']} samples")
    print(f"  Batch size:    {train_loader.batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    print(f"  max_rul:       {preprocess['max_rul']}")
    print("=" * 70)

    # ── training ─────────────────────────────────────────────────────────────
    save_path = str(SAVE_MODEL_DIR)
    os.makedirs(save_path, exist_ok=True)

    models_config = {
        #"lstm": {"lr": 1e-3, "epochs": 160},
        #"gru": {"lr": 1e-3, "epochs": 160},
        #"cnn_lstm": {"lr": 1e-3, "epochs": 160},
        "transformer": {"lr": 5e-4, "epochs": 190},
    }

    histories = {}
    trainers = {}

    for model_type, mc in models_config.items():
        print(f"\n{'#'*80}\n# Training {model_type.upper()}\n{'#'*80}")
        history, trainer = train_single_model(
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            preprocess=preprocess,
            device=device,
            num_epochs=mc["epochs"],
            learning_rate=mc["lr"],
            save_path=save_path,
        )
        histories[model_type] = history
        trainers[model_type] = trainer

    print("\nTraining done. Best checkpoints written with preprocess metadata embedded.")


if __name__ == "__main__":
    main()