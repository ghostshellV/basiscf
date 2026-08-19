import numpy as np
import pandas as pd
from typing import List, Tuple

def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences.
    
    Returns:
        X: shape (n_samples, seq_len, n_features)
        y: shape (n_samples,) — target at the END of each window
    """
    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    
    X_list = []
    y_list = []
    
    n = len(features)
    for i in range(0, n - seq_len + 1, stride):
        X_list.append(features[i: i + seq_len])
        y_list.append(targets[i + seq_len - 1])  # Target at end of window
    
    if len(X_list) == 0:
        return np.empty((0, seq_len, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    return X, y