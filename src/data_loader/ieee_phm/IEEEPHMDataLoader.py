import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
from .FeatureNormalizer import FeatureNormalizer
from .data_utils import create_sequences
from .BearingRULDataset import BearingRULDataset
from .DataConfig import DataConfig
import pickle

def build_dataloaders(
    train_bearing_data: Dict[str, pd.DataFrame],
    test_bearing_data: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    config: DataConfig
) -> Tuple[DataLoader, DataLoader, DataLoader, FeatureNormalizer, Dict]:
    """
    Full pipeline:
    1. Create sequences from all training bearings
    2. Split into train/val (+ optionally use test bearings for test set)
    3. Fit normalizer on train split
    4. Normalize all splits
    5. Create DataLoaders
    
    Returns:
        train_loader, val_loader, test_loader, normalizer, metadata
    """
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    # --- Step 1: Create sequences from training bearings ---
    X_all_train = []
    y_all_train = []
    bearing_indices = []  # track which bearing each sample comes from
    
    for bkey, df_feat in sorted(train_bearing_data.items()):
        X_b, y_b = create_sequences(df_feat, feature_cols, config.target_col, 
                                     config.seq_len, config.stride)
        if len(X_b) > 0:
            X_all_train.append(X_b)
            y_all_train.append(y_b)
            bearing_indices.extend([bkey] * len(X_b))
            print(f"  [TRAIN] {bkey}: {X_b.shape[0]} sequences")
    
    X_train_all = np.concatenate(X_all_train, axis=0)
    y_train_all = np.concatenate(y_all_train, axis=0)
    print(f"\n  Total training sequences: {X_train_all.shape}")
    
    # --- Step 2: Create sequences from test bearings ---
    X_test_all = []
    y_test_all = []
    
    for bkey, df_feat in sorted(test_bearing_data.items()):
        X_b, y_b = create_sequences(df_feat, feature_cols, config.target_col,
                                     config.seq_len, config.stride)
        if len(X_b) > 0:
            X_test_all.append(X_b)
            y_test_all.append(y_b)
            print(f"  [TEST]  {bkey}: {X_b.shape[0]} sequences")
    
    if X_test_all:
        X_test = np.concatenate(X_test_all, axis=0)
        y_test = np.concatenate(y_test_all, axis=0)
    else:
        X_test = np.empty((0, config.seq_len, len(feature_cols)), dtype=np.float32)
        y_test = np.empty((0,), dtype=np.float32)
    
    print(f"  Total test sequences: {X_test.shape}")
    
    # --- Step 3: Split training into train/val ---
    # If we have no separate test bearings, split into train/val/test
    # If we have test bearings, split training into train/val only
    if len(X_test) > 0:
        # Split training bearings into train + val
        val_frac = config.val_ratio / (config.train_ratio + config.val_ratio)
        indices = np.arange(len(X_train_all))
        train_idx, val_idx = train_test_split(
            indices, test_size=val_frac, random_state=config.random_seed
        )
    else:
        # Split into train/val/test from training data
        indices = np.arange(len(X_train_all))
        test_frac = config.test_ratio
        val_frac_of_rest = config.val_ratio / (config.train_ratio + config.val_ratio)
        
        trainval_idx, test_idx = train_test_split(
            indices, test_size=test_frac, random_state=config.random_seed
        )
        train_idx, val_idx = train_test_split(
            trainval_idx, test_size=val_frac_of_rest, random_state=config.random_seed
        )
        X_test = X_train_all[test_idx]
        y_test = y_train_all[test_idx]
    
    X_train = X_train_all[train_idx]
    y_train = y_train_all[train_idx]
    X_val = X_train_all[val_idx]
    y_val = y_train_all[val_idx]
    
    print(f"\n  After split:")
    print(f"    Train:  X={X_train.shape}, y={y_train.shape}")
    print(f"    Val:    X={X_val.shape}, y={y_val.shape}")
    print(f"    Test:   X={X_test.shape}, y={y_test.shape}")
    
    # --- Step 4: Fit normalizer on train split, transform all ---
    normalizer = FeatureNormalizer(scaler_type=config.scaler_type)
    normalizer.fit(X_train, y_train)
    
    X_train_norm = normalizer.transform_X(X_train)
    y_train_norm = normalizer.transform_y(y_train)
    
    X_val_norm = normalizer.transform_X(X_val)
    y_val_norm = normalizer.transform_y(y_val)
    
    X_test_norm = normalizer.transform_X(X_test)
    y_test_norm = normalizer.transform_y(y_test)
    
    print(f"\n  After normalization:")
    print(f"    X_train range: [{X_train_norm.min():.4f}, {X_train_norm.max():.4f}]")
    print(f"    y_train range: [{y_train_norm.min():.4f}, {y_train_norm.max():.4f}]")
    print(f"    X_val range:   [{X_val_norm.min():.4f}, {X_val_norm.max():.4f}]")
    print(f"    X_test range:  [{X_test_norm.min():.4f}, {X_test_norm.max():.4f}]")
    
    # --- Step 5: Create DataLoaders ---
    train_dataset = BearingRULDataset(X_train_norm, y_train_norm)
    val_dataset = BearingRULDataset(X_val_norm, y_val_norm)
    test_dataset = BearingRULDataset(X_test_norm, y_test_norm)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False
    )
    
    # Metadata
    metadata = {
        'n_features': len(feature_cols),
        'seq_len': config.seq_len,
        'feature_cols': feature_cols,
        'n_train': len(train_dataset),
        'n_val': len(val_dataset),
        'n_test': len(test_dataset),
        'scaler_type': config.scaler_type,
        'scaler_params': normalizer.get_params(),
        'y_train_original_range': [float(y_train.min()), float(y_train.max())],
        'y_test_original_range': [float(y_test.min()), float(y_test.max())],
    }
    
    return train_loader, val_loader, test_loader, normalizer, metadata
