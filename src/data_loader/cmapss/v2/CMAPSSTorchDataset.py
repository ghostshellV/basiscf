"""
CMAPSS Torch Dataset & DataLoader.

Replicates the full data processing pipeline from the CiRNN notebooks
(CiRNN_FD001 – CiRNN_FD004) and returns PyTorch DataLoaders.

By default the loader emits  (X, Y)  batches where:
  X : (batch, seq_len, n_features)  – all settings + sensors combined
  Y : (batch,)                      – scalar RUL target

When ``split_context=True``, it emits  (X_primary, Z_context, Y)  instead.

Pipeline:
  1. Load CSV  →  drop unnamed index column
  2. Select feature subset (varies per FD subset)
  3. MinMax normalise all features (cols after unit_number, time_cycles)
  4. [FD002 / FD004 only] KMeans clustering (6 clusters) on normalised
     features  →  per-cluster MinMax normalisation of settings+sensors
  5. Trailing moving-average smoothing (window = 3)
  6. Sliding-window data preparation  →  tensors
  7. Train / val split: last 2×seq_len rows per engine unit go to val
  8. Wrap in TensorDataset + DataLoader
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────── configuration per subset ──────────────────────────

# Feature lists exactly as used in the notebooks
FEATURE_LISTS: Dict[str, List[str]] = {
    "FD001": [
        "unit_number", "time_cycles", "setting_1", "setting_2",
        "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12",
        "s_13", "s_15", "s_17", "s_20", "s_21", "RUL",
    ],
    "FD002": [
        "unit_number", "time_cycles", "setting_1", "setting_2", "setting_3",
        "s_1", "s_2", "s_8", "s_13", "s_14", "s_19", "RUL",
    ],
    "FD003": [
        "unit_number", "time_cycles", "setting_1", "setting_2",
        "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_15",
        "s_17", "s_20", "s_21", "RUL",
    ],
    "FD004": [
        "unit_number", "time_cycles", "setting_1", "setting_2", "setting_3",
        "s_2", "s_8", "s_14", "s_16", "RUL",
    ],
}

# Number of context (settings) columns after unit_number & time_cycles
#   FD001 / FD003 → 2 settings  |  FD002 / FD004 → 3 settings
N_CONTEXT: Dict[str, int] = {
    "FD001": 2, "FD002": 3, "FD003": 2, "FD004": 3,
}

# Whether to apply cluster-based normalisation (multi-operating-condition sets)
USE_CLUSTERING: Dict[str, bool] = {
    "FD001": False, "FD002": True, "FD003": False, "FD004": True,
}

N_CLUSTERS = 6  # KMeans clusters (same across FD002 / FD004)

# Default hyper-params
DEFAULT_SEQ_LEN = 50
DEFAULT_BATCH_SIZE: Dict[str, int] = {
    "FD001": 64, "FD002": 128, "FD003": 128, "FD004": 128,
}

SMOOTHING_WINDOW = 3  # trailing moving-average window

# Default data directory (relative to this file's location)
_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "data", "processed", "CMAPSS", "CiRNN", "RUL Estimation", "data",
)


# ──────────────────────────── helper functions ─────────────────────────────────

def _load_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Drop the unnamed index column that the CSVs carry
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df.drop(columns=df.columns[0], inplace=True)
    return df


def _minmax_transform(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global min-max normalisation. Returns (normalised, min_vals, ranges)."""
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    rng = max_val - min_val
    rng[rng == 0] = 1.0  # avoid division-by-zero
    normed = (data - min_val) / rng
    return normed, min_val, rng


def _inv_minmax(data: np.ndarray, min_val: float, rng: float) -> np.ndarray:
    """Inverse min-max for a 1-D target column."""
    return data * rng + min_val


def _moving_average(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ──────────────── cluster normalisation (FD002 / FD004) ───────────────────────

def _cluster_norm_train(
    df: pd.DataFrame,
    sensor_setting_cols: List[str],
    n_clusters: int = N_CLUSTERS,
) -> Tuple[pd.DataFrame, List[pd.Series], List[pd.Series]]:
    """
    Per-cluster min-max normalisation on the *sensor + setting* columns.
    Returns (normalised_df, list_of_min_per_cluster, list_of_range_per_cluster).
    """
    norm_df = df.copy(deep=True)
    clust_min, clust_range = [], []

    for i in range(n_clusters):
        mask = df["label"] == str(i)
        cluster = df.loc[mask, sensor_setting_cols].copy()
        min_val = cluster.min(axis=0)
        max_val = cluster.max(axis=0)
        range_val = max_val - min_val
        for c in sensor_setting_cols:
            if range_val[c] == 0:
                range_val[c] = 1.0
                min_val[c] = 0.0
        norm_df.loc[mask, sensor_setting_cols] = (cluster - min_val) / range_val
        clust_min.append(min_val)
        clust_range.append(range_val)

    return norm_df, clust_min, clust_range


def _cluster_norm_test(
    df: pd.DataFrame,
    sensor_setting_cols: List[str],
    clust_min: List[pd.Series],
    clust_range: List[pd.Series],
    n_clusters: int = N_CLUSTERS,
) -> pd.DataFrame:
    """Apply cluster normalisation statistics from *train* to *test* data."""
    out = df.copy(deep=True)
    for i in range(n_clusters):
        mask = df["label"] == i  # test labels are ints (from kmeans.predict)
        cluster = df.loc[mask, sensor_setting_cols].copy()
        out.loc[mask, sensor_setting_cols] = (
            (cluster - clust_min[i]) / clust_range[i]
        )
    return out


# ────────────────────── smoothing per engine unit ──────────────────────────────

def _smooth_array(
    data: np.ndarray,
    unit_col: int,
    feature_cols_range: Tuple[int, int],
    rul_col: int,
    w: int = SMOOTHING_WINDOW,
    label_col: Optional[int] = None,
) -> np.ndarray:
    """
    Apply trailing moving-average smoothing separately per engine unit.

    Parameters
    ----------
    data : (N, M) array where columns follow the feature_list order.
    unit_col : column index for unit_number (always 0).
    feature_cols_range : (start, end) column indices to smooth (settings + sensors).
    rul_col : column index for RUL.
    w : smoothing window size.
    label_col : optional column index for cluster label (kept unsmoothed).
    """
    start, end = feature_cols_range
    units = np.unique(data[:, unit_col]).astype(int)
    chunks = []

    for uid in units:
        grp = data[data[:, unit_col] == uid]
        n = grp.shape[0]
        if n < w:
            continue  # skip very short sequences
        m = grp.shape[1]
        smoothed = np.zeros((n - w + 1, m))
        # Copy passthrough columns
        smoothed[:, 0:2] = grp[w - 1 :, 0:2]    # unit_number, time_cycles
        smoothed[:, rul_col] = grp[w - 1 :, rul_col]  # RUL
        if label_col is not None:
            smoothed[:, label_col] = grp[w - 1 :, label_col]
        # Smooth features
        for j in range(start, end):
            smoothed[:, j] = _moving_average(grp[:, j], w)
        chunks.append(smoothed)

    return np.concatenate(chunks, axis=0)


# ──────────────────── sliding-window data preparation ──────────────────────────

def _data_preparation(
    data: np.ndarray,
    n_past: int,
    n_context: int,
    split_context: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Prepare sliding-window samples (sequence-to-value).

    Columns layout: [unit_number, time_cycles, <context>, <primary_sensors>, RUL]

    When *split_context* is False (default):
        X contains ALL features (settings + sensors)  →  (N, n_past, n_all_features)
        Z is None
    When *split_context* is True:
        X contains only primary sensors                →  (N, n_past, n_primary)
        Z contains context (settings)                  →  (N, n_past, n_context)

    Returns
    -------
    U : (N, n_past, 2)  – unit_number, time_cycles
    X : (N, n_past, n_features) – input features
    Y : (N,) – scalar RUL target at end of window
    Z : (N, n_past, n_context) or None – context features (only when split_context=True)
    """
    n, m = data.shape
    ctx_start = 2
    ctx_end = 2 + n_context
    sensor_end = m - 1  # last col is RUL

    U_list, X_list, Y_list, Z_list = [], [], [], []

    for i in range(n_past, n):
        U_list.append(data[i - n_past : i, 0:2])
        if split_context:
            Z_list.append(data[i - n_past : i, ctx_start:ctx_end])
            X_list.append(data[i - n_past : i, ctx_end:sensor_end])
        else:
            X_list.append(data[i - n_past : i, ctx_start:sensor_end])
        Y_list.append(data[i, sensor_end])

    return (
        np.array(U_list),
        np.array(X_list),
        np.array(Y_list),
        np.array(Z_list) if split_context else None,
    )


# ──────────────────────────── main dataclass ───────────────────────────────────

@dataclass
class CMAPSSPreprocessInfo:
    """Stores all information needed for inverse transforms at inference time."""
    subset: str
    feature_list: List[str]
    n_context: int
    n_features: int  # total input features in X (depends on split_context)
    seq_len: int
    batch_size: int
    smoothing_window: int
    use_clustering: bool
    split_context: bool
    # Global min-max params (for columns [2:] of feature_list)
    global_min: np.ndarray = field(repr=False)
    global_range: np.ndarray = field(repr=False)
    # Cluster params (only populated when use_clustering is True)
    cluster_min: Optional[List[pd.Series]] = field(default=None, repr=False)
    cluster_range: Optional[List[pd.Series]] = field(default=None, repr=False)
    kmeans_model: Optional[KMeans] = field(default=None, repr=False)

    @property
    def rul_min(self) -> float:
        return float(self.global_min[-1])

    @property
    def rul_range(self) -> float:
        return float(self.global_range[-1])

    def inverse_transform_rul(self, normalised_rul: np.ndarray) -> np.ndarray:
        """De-normalise RUL predictions back to original scale (1-level)."""
        return _inv_minmax(normalised_rul, self.rul_min, self.rul_range)


class CMAPSSTorchDataset:
    """
    End-to-end data loader that mirrors the CiRNN notebook pipelines.

    Usage (default – no context split)
    ----------------------------------
    >>> ds = CMAPSSTorchDataset(data_dir="/path/to/data")
    >>> train_loader, val_loader, test_loader, info = ds.load("FD001")
    >>> for X, Y in train_loader:
    ...     # X: (batch, seq_len, n_features)  – all settings + sensors
    ...     # Y: (batch,)                      – scalar RUL
    ...     pass

    Usage (with context split)
    --------------------------
    >>> train_loader, val_loader, test_loader, info = ds.load("FD001", split_context=True)
    >>> for X, Z, Y in train_loader:
    ...     # X: (batch, seq_len, n_primary)
    ...     # Z: (batch, seq_len, n_context)
    ...     # Y: (batch,)
    ...     pass
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = str(Path(data_dir or _DEFAULT_DATA_DIR).resolve())

    # ─────────────────────── public API ────────────────────────

    def load(
        self,
        subset: str = "FD001",
        seq_len: int = DEFAULT_SEQ_LEN,
        batch_size: Optional[int] = None,
        smoothing_window: int = SMOOTHING_WINDOW,
        split_context: bool = False,
        seed: int = 40,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, CMAPSSPreprocessInfo]:
        """
        Load & preprocess a CMAPSS FD subset, returning train/val/test loaders.

        Parameters
        ----------
        subset : one of "FD001", "FD002", "FD003", "FD004".
        seq_len : sliding-window length (default 50).
        batch_size : DataLoader batch size. Defaults to notebook value.
        smoothing_window : trailing moving-average window (default 3).
        split_context : if True, emit (X_primary, Z_context, Y); if False
            (default), emit (X, Y) where X contains *all* features.
        seed : random seed for reproducibility.

        Returns
        -------
        train_loader, val_loader, test_loader, preprocess_info
        """
        assert subset in FEATURE_LISTS, f"Unknown subset: {subset}"

        batch_size = batch_size or DEFAULT_BATCH_SIZE[subset]
        feature_list = FEATURE_LISTS[subset]
        n_ctx = N_CONTEXT[subset]
        clustering = USE_CLUSTERING[subset]

        # ── 1. Load CSVs ──────────────────────────────────────────
        train_df = _load_csv(os.path.join(self.data_dir, f"train_{subset}"))
        test_df = _load_csv(os.path.join(self.data_dir, f"test_{subset}"))

        # ── 2. Select features ─────────────────────────────────────
        train_df = train_df[feature_list].copy()
        test_df = test_df[feature_list].copy()

        # ── 3. Global min-max normalisation ────────────────────────
        train_vals = np.array(train_df.iloc[:, 2:])  # drop unit_number, time_cycles
        normed_train, p_min, p_range = _minmax_transform(train_vals)

        # Settings + sensor col names (everything between time_cycles and RUL)
        settings_sensor_cols = feature_list[2:-1]  # excludes RUL
        all_normed_cols = feature_list[2:]          # includes RUL

        if clustering:
            # ── 4a. KMeans clustering (FD002 / FD004) ─────────────
            kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=0)
            kmeans.fit(normed_train)

            # Build clustered train DF
            train_clust = train_df[["unit_number", "time_cycles"]].copy()
            train_clust[all_normed_cols] = pd.DataFrame(
                normed_train, columns=all_normed_cols, index=train_clust.index,
            )
            train_clust["label"] = kmeans.labels_.astype(str)

            # Per-cluster normalisation on train
            train_norm_df, clust_min, clust_range = _cluster_norm_train(
                train_clust, settings_sensor_cols, N_CLUSTERS,
            )

            # ── Test: global norm  →  predict labels  →  cluster norm
            test_normed = (test_df[all_normed_cols] - p_min) / p_range
            test_labels = kmeans.predict(test_normed.values)
            test_clust = test_df[["unit_number", "time_cycles"]].copy()
            test_clust[all_normed_cols] = test_normed.values
            test_clust["label"] = test_labels
            test_norm_df = _cluster_norm_test(
                test_clust, settings_sensor_cols, clust_min, clust_range, N_CLUSTERS,
            )

            # Convert to arrays for smoothing; drop label column
            train_arr = train_norm_df[feature_list].to_numpy()
            test_arr = test_norm_df[feature_list].to_numpy()

        else:
            # ── 4b. No clustering (FD001 / FD003) ─────────────────
            kmeans = None
            clust_min = None
            clust_range = None

            # Rebuild array with unit_number + time_cycles + normalised features
            train_arr = np.column_stack([
                train_df[["unit_number", "time_cycles"]].values,
                normed_train,
            ])

            # Normalise test using train statistics
            test_vals = np.array(test_df.iloc[:, 2:])
            test_normed = (test_vals - p_min) / p_range
            test_arr = np.column_stack([
                test_df[["unit_number", "time_cycles"]].values,
                test_normed,
            ])

        # ── 5. Smoothing ──────────────────────────────────────────
        n_features = len(feature_list)
        rul_col = n_features - 1
        feat_range = (2, rul_col)  # settings + sensors to smooth

        smooth_train = _smooth_array(
            train_arr, unit_col=0, feature_cols_range=feat_range,
            rul_col=rul_col, w=smoothing_window,
        )
        smooth_test = _smooth_array(
            test_arr, unit_col=0, feature_cols_range=feat_range,
            rul_col=rul_col, w=smoothing_window,
        )

        # ── 6. Train / Val split ──────────────────────────────────
        #   Last 2×seq_len rows per engine unit go to validation
        unit_ids = np.unique(smooth_train[:, 0]).astype(int)
        train_chunks, val_chunks = [], []
        for uid in unit_ids:
            eng = smooth_train[smooth_train[:, 0] == uid]
            if len(eng) > 2 * seq_len:
                train_chunks.append(eng[: -2 * seq_len])
                val_chunks.append(eng[-2 * seq_len :])
            else:
                # If engine has too few rows, put all in train
                train_chunks.append(eng)

        data_train = np.concatenate(train_chunks, axis=0)
        data_val = np.concatenate(val_chunks, axis=0) if val_chunks else np.empty((0, n_features))

        # ── 7. Sliding-window preparation ─────────────────────────
        U_train, X_train, Y_train, Z_train = self._prepare_per_unit(
            data_train, seq_len, n_ctx, split_context,
        )
        U_val, X_val, Y_val, Z_val = self._prepare_per_unit(
            data_val, seq_len, n_ctx, split_context,
        )
        U_test, X_test, Y_test, Z_test = self._prepare_per_unit(
            smooth_test, seq_len, n_ctx, split_context,
        )

        # ── 8. Build DataLoaders ──────────────────────────────────
        train_loader = self._make_loader(X_train, Z_train, Y_train, batch_size, shuffle=False)
        val_loader = self._make_loader(X_val, Z_val, Y_val, batch_size, shuffle=False)
        test_loader = self._make_loader(X_test, Z_test, Y_test, batch_size=1, shuffle=False)

        n_feat = X_train.shape[2] if len(X_train) > 0 else 0
        info = CMAPSSPreprocessInfo(
            subset=subset,
            feature_list=feature_list,
            n_context=n_ctx,
            n_features=n_feat,
            seq_len=seq_len,
            batch_size=batch_size,
            smoothing_window=smoothing_window,
            use_clustering=clustering,
            split_context=split_context,
            global_min=p_min,
            global_range=p_range,
            cluster_min=clust_min,
            cluster_range=clust_range,
            kmeans_model=kmeans,
        )

        print(f"[{subset}] train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
              f"| seq_len={seq_len}, n_features={n_feat}, n_context={n_ctx}, "
              f"split_context={split_context}, batch={batch_size}")

        return train_loader, val_loader, test_loader, info

    # ────────────────── internal helpers ───────────────────────

    @staticmethod
    def _prepare_per_unit(
        data: np.ndarray,
        seq_len: int,
        n_context: int,
        split_context: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run sliding-window preparation per engine unit, then concatenate."""
        if data.shape[0] == 0:
            m = data.shape[1] if data.ndim == 2 else 0
            n_all = m - 2 - 1  # cols - unit/time - RUL
            if split_context:
                n_x = n_all - n_context
            else:
                n_x = n_all
            return (
                np.empty((0, seq_len, 2)),
                np.empty((0, seq_len, max(n_x, 0))),
                np.empty((0,)),
                np.empty((0, seq_len, n_context)) if split_context else None,
            )

        unit_ids = np.unique(data[:, 0]).astype(int)
        all_U, all_X, all_Y, all_Z = [], [], [], []

        for uid in unit_ids:
            grp = data[data[:, 0] == uid]
            if grp.shape[0] < seq_len + 1:
                continue
            U, X, Y, Z = _data_preparation(grp, seq_len, n_context, split_context)
            all_U.append(U)
            all_X.append(X)
            all_Y.append(Y)
            if Z is not None:
                all_Z.append(Z)

        if not all_X:
            m = data.shape[1]
            n_all = m - 2 - 1
            if split_context:
                n_x = n_all - n_context
            else:
                n_x = n_all
            return (
                np.empty((0, seq_len, 2)),
                np.empty((0, seq_len, max(n_x, 0))),
                np.empty((0,)),
                np.empty((0, seq_len, n_context)) if split_context else None,
            )

        return (
            np.concatenate(all_U, axis=0),
            np.concatenate(all_X, axis=0),
            np.concatenate(all_Y, axis=0),
            np.concatenate(all_Z, axis=0) if all_Z else None,
        )

    @staticmethod
    def _make_loader(
        X: np.ndarray,
        Z: Optional[np.ndarray],
        Y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
    ) -> DataLoader:
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(Y, dtype=torch.float32)  # (N,)
        if Z is not None:
            z_tensor = torch.tensor(Z, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, z_tensor, y_tensor)
        else:
            dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=(batch_size > 1),
        )
