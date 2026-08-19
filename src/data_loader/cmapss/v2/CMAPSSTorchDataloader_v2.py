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


# -----------------------------------------------------------------------------
# Feature definitions
# -----------------------------------------------------------------------------

_ALL_COLUMNS = [
    "unit_number",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"s_{i}" for i in range(1, 22)],
]

FEATURE_LISTS: Dict[str, List[str]] = {
    # Kept for backward compatibility with the current training code.
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

FULL_FEATURE_LISTS: Dict[str, List[str]] = {
    fd: _ALL_COLUMNS + ["RUL"] for fd in ("FD001", "FD002", "FD003", "FD004")
}

N_CONTEXT: Dict[str, int] = {
    "FD001": 2, "FD002": 3, "FD003": 2, "FD004": 3,
}

USE_CLUSTERING: Dict[str, bool] = {
    "FD001": False, "FD002": True, "FD003": False, "FD004": True,
}

DEFAULT_SEQ_LEN = 50
DEFAULT_BATCH_SIZE: Dict[str, int] = {
    "FD001": 64, "FD002": 128, "FD003": 128, "FD004": 128,
}
SMOOTHING_WINDOW = 3
N_CLUSTERS = 6

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "data", "processed", "CMAPSS", "CiRNN", "RUL Estimation", "data",
)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


class MinMaxScalerNP:
    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "MinMaxScalerNP":
        x = _ensure_2d(np.asarray(x, dtype=np.float32))
        self.min_ = x.min(axis=0)
        max_ = x.max(axis=0)
        rng = max_ - self.min_
        rng[rng == 0.0] = 1.0
        self.range_ = rng
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler must be fit before transform().")
        x = _ensure_2d(np.asarray(x, dtype=np.float32))
        return (x - self.min_) / self.range_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler must be fit before inverse_transform().")
        x = _ensure_2d(np.asarray(x, dtype=np.float32))
        return x * self.range_ + self.min_



def _find_existing_file(data_dir: str, stem: str) -> Optional[str]:
    candidates = [
        stem,
        f"{stem}.csv",
        f"{stem}.txt",
        stem.lower(),
        f"{stem.lower()}.csv",
        f"{stem.lower()}.txt",
    ]
    for cand in candidates:
        p = os.path.join(data_dir, cand)
        if os.path.exists(p):
            return p
    return None



def _load_table(filepath: str) -> pd.DataFrame:
    # Try CSV-style first.
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] > 1:
            return _cleanup_dataframe(df)
    except Exception:
        pass

    # Then try whitespace-delimited NASA txt format.
    df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    return _cleanup_dataframe(df)



def _cleanup_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop fully empty columns and unnamed columns.
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    drop_cols = []
    for c in df.columns:
        cs = str(c).strip().lower()
        if cs.startswith("unnamed") or cs == "":
            drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # NASA raw files have 26 columns without headers.
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        if df.shape[1] >= 26:
            df = df.iloc[:, :26].copy()
            df.columns = _ALL_COLUMNS
        elif df.shape[1] == 1:
            raise ValueError(f"Could not parse data file correctly; got shape={df.shape}")

    # Normalise column names.
    rename_map = {}
    for c in df.columns:
        cs = str(c).strip()
        rename_map[c] = cs
    df = df.rename(columns=rename_map)

    return df



def _load_truth_vector(data_dir: str, subset: str) -> Optional[np.ndarray]:
    path = _find_existing_file(data_dir, f"RUL_{subset}")
    if path is None:
        return None
    df = _load_table(path)
    vals = df.to_numpy().reshape(-1)
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=np.float32)
    return vals if vals.size > 0 else None



def _compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    max_cycles = out.groupby("unit_number")["time_cycles"].transform("max")
    out["RUL"] = (max_cycles - out["time_cycles"]).astype(np.float32)
    return out



def _compute_test_rul_from_truth(df: pd.DataFrame, rul_vector: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    last_cycles = out.groupby("unit_number")["time_cycles"].transform("max")
    # Official RUL file provides one extra-life value per test engine (1-indexed by unit number).
    unit_ids = out["unit_number"].astype(int).to_numpy()
    extra = np.array([rul_vector[u - 1] for u in unit_ids], dtype=np.float32)
    out["RUL"] = (last_cycles - out["time_cycles"]).to_numpy(dtype=np.float32) + extra
    return out



def _apply_piecewise_cap(rul: pd.Series, cap: Optional[int]) -> pd.Series:
    if cap is None:
        return rul.astype(np.float32)
    return np.minimum(rul.to_numpy(dtype=np.float32), float(cap)).astype(np.float32)



def _rolling_mean_per_unit(df: pd.DataFrame, cols: List[str], window: int) -> pd.DataFrame:
    if window <= 1:
        return df.copy()

    chunks = []
    for _, grp in df.groupby("unit_number", sort=True):
        grp = grp.sort_values("time_cycles").copy()
        if len(grp) < window:
            continue
        sm = grp.copy()
        sm[cols] = grp[cols].rolling(window=window, min_periods=window).mean()
        sm = sm.iloc[window - 1 :].reset_index(drop=True)
        chunks.append(sm)

    if not chunks:
        return df.iloc[0:0].copy()
    return pd.concat(chunks, axis=0, ignore_index=True)



def _pad_left(arr: np.ndarray, target_len: int) -> np.ndarray:
    if arr.shape[0] >= target_len:
        return arr[-target_len:]
    pad_len = target_len - arr.shape[0]
    pad = np.repeat(arr[:1], repeats=pad_len, axis=0)
    return np.concatenate([pad, arr], axis=0)



def _windows_from_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    seq_len: int,
    split_context: bool,
    n_context: int,
    last_window_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if df.empty:
        n_total = len(feature_cols)
        n_primary = n_total - n_context if split_context else n_total
        return (
            np.empty((0, seq_len, 2), dtype=np.float32),
            np.empty((0, seq_len, max(n_primary, 0)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, seq_len, n_context), dtype=np.float32) if split_context else None,
        )

    context_cols = feature_cols[:n_context]
    primary_cols = feature_cols[n_context:]

    U_list: List[np.ndarray] = []
    X_list: List[np.ndarray] = []
    Y_list: List[float] = []
    Z_list: List[np.ndarray] = []

    for _, grp in df.groupby("unit_number", sort=True):
        grp = grp.sort_values("time_cycles")
        meta = grp[["unit_number", "time_cycles"]].to_numpy(dtype=np.float32)
        feats = grp[feature_cols].to_numpy(dtype=np.float32)
        target = grp[target_col].to_numpy(dtype=np.float32)

        if last_window_only:
            feat_win = _pad_left(feats, seq_len)
            meta_win = _pad_left(meta, seq_len)
            U_list.append(meta_win)
            if split_context:
                Z_list.append(feat_win[:, :n_context])
                X_list.append(feat_win[:, n_context:])
            else:
                X_list.append(feat_win)
            Y_list.append(float(target[-1]))
            continue

        if len(grp) < seq_len + 1:
            continue

        for end in range(seq_len, len(grp)):
            start = end - seq_len
            feat_win = feats[start:end]
            meta_win = meta[start:end]
            U_list.append(meta_win)
            if split_context:
                Z_list.append(feat_win[:, :n_context])
                X_list.append(feat_win[:, n_context:])
            else:
                X_list.append(feat_win)
            Y_list.append(float(target[end]))

    if not X_list:
        n_total = len(feature_cols)
        n_primary = n_total - n_context if split_context else n_total
        return (
            np.empty((0, seq_len, 2), dtype=np.float32),
            np.empty((0, seq_len, max(n_primary, 0)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, seq_len, n_context), dtype=np.float32) if split_context else None,
        )

    U = np.stack(U_list).astype(np.float32)
    X = np.stack(X_list).astype(np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    Z = np.stack(Z_list).astype(np.float32) if split_context else None
    return U, X, Y, Z



def _cluster_condition_normalise(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    operating_cols: List[str],
    sensor_cols: List[str],
    n_clusters: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans, List[pd.Series], List[pd.Series]]:
    # IMPORTANT: cluster only on operating conditions, never on RUL.
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    train_labels = kmeans.fit_predict(train_df[operating_cols].to_numpy(dtype=np.float32))
    test_labels = kmeans.predict(test_df[operating_cols].to_numpy(dtype=np.float32))

    tr = train_df.copy()
    te = test_df.copy()
    tr["cluster_id"] = train_labels
    te["cluster_id"] = test_labels

    cluster_min: List[pd.Series] = []
    cluster_range: List[pd.Series] = []

    for cid in range(n_clusters):
        mask_tr = tr["cluster_id"] == cid
        if not mask_tr.any():
            min_s = pd.Series(0.0, index=sensor_cols)
            rng_s = pd.Series(1.0, index=sensor_cols)
        else:
            cluster_block = tr.loc[mask_tr, sensor_cols]
            min_s = cluster_block.min(axis=0)
            max_s = cluster_block.max(axis=0)
            rng_s = max_s - min_s
            rng_s[rng_s == 0.0] = 1.0
            tr.loc[mask_tr, sensor_cols] = (cluster_block - min_s) / rng_s

        mask_te = te["cluster_id"] == cid
        if mask_te.any():
            te.loc[mask_te, sensor_cols] = (te.loc[mask_te, sensor_cols] - min_s) / rng_s

        cluster_min.append(min_s)
        cluster_range.append(rng_s)

    tr = tr.drop(columns=["cluster_id"])
    te = te.drop(columns=["cluster_id"])
    return tr, te, kmeans, cluster_min, cluster_range


# -----------------------------------------------------------------------------
# Metadata dataclass
# -----------------------------------------------------------------------------


@dataclass
class CMAPSSPreprocessInfo:
    subset: str
    feature_list: List[str]
    n_context: int
    n_features: int
    seq_len: int
    batch_size: int
    smoothing_window: int
    use_clustering: bool
    split_context: bool
    feature_mode: str = "selected"
    piecewise_rul_cap: Optional[int] = 125
    val_strategy: str = "engine"
    test_last_window_only: bool = True
    global_min: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    global_range: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    cluster_min: Optional[List[pd.Series]] = field(default=None, repr=False)
    cluster_range: Optional[List[pd.Series]] = field(default=None, repr=False)
    kmeans_model: Optional[KMeans] = field(default=None, repr=False)
    train_unit_ids: Optional[List[int]] = None
    val_unit_ids: Optional[List[int]] = None

    @property
    def rul_min(self) -> float:
        return float(self.global_min[-1]) if self.global_min.size else 0.0

    @property
    def rul_range(self) -> float:
        return float(self.global_range[-1]) if self.global_range.size else 1.0

    def inverse_transform_rul(self, normalised_rul: np.ndarray) -> np.ndarray:
        arr = np.asarray(normalised_rul, dtype=np.float32)
        return arr * self.rul_range + self.rul_min


# -----------------------------------------------------------------------------
# Main dataset class
# -----------------------------------------------------------------------------


class CMAPSSTorchDataset:
    """
    Improved CMAPSS dataset loader.

    Main changes relative to the uploaded version:
      - no target leakage in clustering
      - benchmark-friendly test loader (one last window per engine by default)
      - optional piecewise RUL cap (default 125)
      - engine-level validation split by default
      - shuffling enabled for train loader
      - no drop_last for val/test
      - flexible reading of both processed CSV files and original NASA txt files
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = str(Path(data_dir or _DEFAULT_DATA_DIR).resolve())

    def load(
        self,
        subset: str = "FD001",
        seq_len: int = DEFAULT_SEQ_LEN,
        batch_size: Optional[int] = None,
        smoothing_window: int = SMOOTHING_WINDOW,
        split_context: bool = False,
        seed: int = 40,
        feature_mode: str = "selected",          # "selected" or "full"
        piecewise_rul_cap: Optional[int] = 125,
        val_strategy: str = "engine",            # "engine" or "tail"
        val_fraction: float = 0.20,
        val_tail_multiple: int = 2,
        test_last_window_only: bool = True,
        shuffle_train: bool = True,
        target_normalisation: bool = True,
        cluster_by_operating_condition: bool = True,
        n_clusters: int = N_CLUSTERS,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_constant_features: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, CMAPSSPreprocessInfo]:
        assert subset in FEATURE_LISTS, f"Unknown subset: {subset}"
        assert feature_mode in {"selected", "full"}
        assert val_strategy in {"engine", "tail"}

        rng = np.random.default_rng(seed)
        batch_size = batch_size or DEFAULT_BATCH_SIZE[subset]

        feature_list = FEATURE_LISTS[subset] if feature_mode == "selected" else FULL_FEATURE_LISTS[subset]
        n_ctx = N_CONTEXT[subset]
        use_clustering = USE_CLUSTERING[subset] and cluster_by_operating_condition

        train_path = _find_existing_file(self.data_dir, f"train_{subset}")
        test_path = _find_existing_file(self.data_dir, f"test_{subset}")
        if train_path is None or test_path is None:
            raise FileNotFoundError(
                f"Could not find train/test files for {subset} in {self.data_dir}. "
                f"Expected files like train_{subset}, train_{subset}.csv, test_{subset}.txt, etc."
            )

        train_df = _load_table(train_path)
        test_df = _load_table(test_path)
        rul_truth = _load_truth_vector(self.data_dir, subset)

        # Add RUL if missing.
        if "RUL" not in train_df.columns:
            train_df = _compute_train_rul(train_df)
        if "RUL" not in test_df.columns:
            if rul_truth is None:
                raise ValueError(
                    f"Test file for {subset} has no RUL column and no RUL_{subset} truth file was found."
                )
            test_df = _compute_test_rul_from_truth(test_df, rul_truth)

        # Piecewise/capped RUL is widely used on CMAPSS.
        train_df["RUL"] = _apply_piecewise_cap(train_df["RUL"], piecewise_rul_cap)
        test_df["RUL"] = _apply_piecewise_cap(test_df["RUL"], piecewise_rul_cap)

        # Select features.
        keep_cols = feature_list.copy()
        for base_col in ["unit_number", "time_cycles", "RUL"]:
            if base_col not in keep_cols:
                keep_cols.append(base_col)
        train_df = train_df[keep_cols].copy()
        test_df = test_df[keep_cols].copy()

        input_cols = [c for c in feature_list if c not in {"unit_number", "time_cycles", "RUL"}]
        operating_cols = [c for c in input_cols[:n_ctx] if c.startswith("setting_")]
        sensor_cols = [c for c in input_cols if c not in operating_cols]

        if drop_constant_features:
            nunique = train_df[input_cols].nunique(dropna=False)
            active_cols = nunique[nunique > 1].index.tolist()
            input_cols = active_cols
            feature_list = ["unit_number", "time_cycles", *input_cols, "RUL"]
            n_ctx = sum(c.startswith("setting_") for c in input_cols)
            operating_cols = [c for c in input_cols if c.startswith("setting_")]
            sensor_cols = [c for c in input_cols if c not in operating_cols]
            train_df = train_df[["unit_number", "time_cycles", *input_cols, "RUL"]].copy()
            test_df = test_df[["unit_number", "time_cycles", *input_cols, "RUL"]].copy()

        # Fit scalers using TRAIN ONLY.
        # Cast input columns to float32 first so pandas accepts the transformed values.
        train_df[input_cols] = train_df[input_cols].astype(np.float32)
        test_df[input_cols] = test_df[input_cols].astype(np.float32)
        input_scaler = MinMaxScalerNP().fit(train_df[input_cols].to_numpy(dtype=np.float32))
        train_df.loc[:, input_cols] = input_scaler.transform(train_df[input_cols].to_numpy(dtype=np.float32))
        test_df.loc[:, input_cols] = input_scaler.transform(test_df[input_cols].to_numpy(dtype=np.float32))

        train_df["RUL"] = train_df["RUL"].astype(np.float32)
        test_df["RUL"] = test_df["RUL"].astype(np.float32)
        target_scaler = MinMaxScalerNP().fit(train_df[["RUL"]].to_numpy(dtype=np.float32))
        if target_normalisation:
            train_df.loc[:, "RUL"] = target_scaler.transform(train_df[["RUL"]].to_numpy(dtype=np.float32)).reshape(-1)
            test_df.loc[:, "RUL"] = target_scaler.transform(test_df[["RUL"]].to_numpy(dtype=np.float32)).reshape(-1)
        else:
            target_scaler.min_ = np.array([0.0], dtype=np.float32)
            target_scaler.range_ = np.array([1.0], dtype=np.float32)

        # Optional operating-condition normalisation for FD002/FD004.
        cluster_min = None
        cluster_range = None
        kmeans = None
        if use_clustering and len(operating_cols) > 0 and len(sensor_cols) > 0:
            train_df, test_df, kmeans, cluster_min, cluster_range = _cluster_condition_normalise(
                train_df=train_df,
                test_df=test_df,
                operating_cols=operating_cols,
                sensor_cols=sensor_cols,
                n_clusters=n_clusters,
                seed=seed,
            )

        # Smoothing is applied ONLY to inputs, never to target.
        train_df = _rolling_mean_per_unit(train_df, cols=input_cols, window=smoothing_window)
        test_df = _rolling_mean_per_unit(test_df, cols=input_cols, window=smoothing_window)

        # Validation split.
        unit_ids = np.array(sorted(train_df["unit_number"].unique().tolist()), dtype=int)
        train_units: List[int]
        val_units: List[int]

        if val_strategy == "engine":
            n_val = max(1, int(round(val_fraction * len(unit_ids))))
            perm = rng.permutation(unit_ids)
            val_units = sorted(perm[:n_val].tolist())
            train_units = sorted(perm[n_val:].tolist())
            df_train_split = train_df[train_df["unit_number"].isin(train_units)].copy()
            df_val_split = train_df[train_df["unit_number"].isin(val_units)].copy()
        else:
            train_chunks = []
            val_chunks = []
            train_units = unit_ids.tolist()
            val_units = unit_ids.tolist()
            tail = val_tail_multiple * seq_len
            for _, grp in train_df.groupby("unit_number", sort=True):
                grp = grp.sort_values("time_cycles")
                if len(grp) > tail:
                    train_chunks.append(grp.iloc[:-tail].copy())
                    val_chunks.append(grp.iloc[-tail:].copy())
                else:
                    train_chunks.append(grp.copy())
            df_train_split = pd.concat(train_chunks, axis=0, ignore_index=True) if train_chunks else train_df.iloc[0:0].copy()
            df_val_split = pd.concat(val_chunks, axis=0, ignore_index=True) if val_chunks else train_df.iloc[0:0].copy()

        # Build windows.
        # _, X_train, Y_train, Z_train = _windows_from_dataframe(
        #     df_train_split,
        #     feature_cols=input_cols,
        #     target_col="RUL",
        #     seq_len=seq_len,
        #     split_context=split_context,
        #     n_context=n_ctx,
        #     last_window_only=False,
        # )
        # _, X_val, Y_val, Z_val = _windows_from_dataframe(
        #     df_val_split,
        #     feature_cols=input_cols,
        #     target_col="RUL",
        #     seq_len=seq_len,
        #     split_context=split_context,
        #     n_context=n_ctx,
        #     last_window_only=False,
        # )
        # _, X_test, Y_test, Z_test = _windows_from_dataframe(
        #     test_df,
        #     feature_cols=input_cols,
        #     target_col="RUL",
        #     seq_len=seq_len,
        #     split_context=split_context,
        #     n_context=n_ctx,
        #     last_window_only=test_last_window_only,
        # )
        U_train, X_train, Y_train, Z_train = _windows_from_dataframe(
            df_train_split,
            feature_cols=input_cols,
            target_col="RUL",
            seq_len=seq_len,
            split_context=split_context,
            n_context=n_ctx,
            last_window_only=False,
        )
        U_val, X_val, Y_val, Z_val = _windows_from_dataframe(
            df_val_split,
            feature_cols=input_cols,
            target_col="RUL",
            seq_len=seq_len,
            split_context=split_context,
            n_context=n_ctx,
            last_window_only=False,
        )
        U_test, X_test, Y_test, Z_test = _windows_from_dataframe(
            test_df,
            feature_cols=input_cols,
            target_col="RUL",
            seq_len=seq_len,
            split_context=split_context,
            n_context=n_ctx,
            last_window_only=test_last_window_only,
        )

        train_loader = self._make_loader(
            X=X_train,
            Z=Z_train,
            Y=Y_train,
            batch_size=batch_size,
            shuffle=shuffle_train,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = self._make_loader(
            X=X_val,
            Z=Z_val,
            Y=Y_val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = self._make_loader(
            X=X_test,
            Z=Z_test,
            Y=Y_test,
            batch_size=1 if test_last_window_only else batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        input_min = np.asarray(input_scaler.min_, dtype=np.float32)
        input_range = np.asarray(input_scaler.range_, dtype=np.float32)
        rul_min = float(target_scaler.min_[0]) if target_scaler.min_ is not None else 0.0
        rul_range = float(target_scaler.range_[0]) if target_scaler.range_ is not None else 1.0
        global_min = np.concatenate([input_min, np.array([rul_min], dtype=np.float32)])
        global_range = np.concatenate([input_range, np.array([rul_range], dtype=np.float32)])

        n_feat = int(X_train.shape[2]) if X_train.ndim == 3 and X_train.shape[0] > 0 else (
            len(input_cols) - n_ctx if split_context else len(input_cols)
        )

        info = CMAPSSPreprocessInfo(
            subset=subset,
            feature_list=["unit_number", "time_cycles", *input_cols, "RUL"],
            n_context=n_ctx,
            n_features=n_feat,
            seq_len=seq_len,
            batch_size=batch_size,
            smoothing_window=smoothing_window,
            use_clustering=use_clustering,
            split_context=split_context,
            feature_mode=feature_mode,
            piecewise_rul_cap=piecewise_rul_cap,
            val_strategy=val_strategy,
            test_last_window_only=test_last_window_only,
            global_min=global_min,
            global_range=global_range,
            cluster_min=cluster_min,
            cluster_range=cluster_range,
            kmeans_model=kmeans,
            train_unit_ids=train_units,
            val_unit_ids=val_units,
        )

        # ------------------------------------------------------------------
        # Store metadata for downstream analysis / plotting
        # ------------------------------------------------------------------
        self.last_subset = subset
        self.last_feature_list = ["unit_number", "time_cycles", *input_cols, "RUL"]
        self.last_input_cols = input_cols.copy()
        self.last_seq_len = seq_len

        # processed full trajectories after scaling/smoothing
        self.last_train_processed_df = train_df.copy()
        self.last_test_processed_df = test_df.copy()

        # window-level metadata: U = [unit_number, time_cycles]
        self.last_U_train = U_train.copy()
        self.last_U_val = U_val.copy()
        self.last_U_test = U_test.copy()

        # optional cached arrays for debugging
        self.last_X_train = X_train.copy()
        self.last_X_val = X_val.copy()
        self.last_X_test = X_test.copy()
        self.last_Y_train = Y_train.copy()
        self.last_Y_val = Y_val.copy()
        self.last_Y_test = Y_test.copy()

        print(
            f"[{subset}] train={len(X_train)}, val={len(X_val)}, test={len(X_test)} | "
            f"seq_len={seq_len}, n_features={n_feat}, feature_mode={feature_mode}, "
            f"val_strategy={val_strategy}, cap={piecewise_rul_cap}, batch={batch_size}, "
            f"test_last_window_only={test_last_window_only}"
        )

        return train_loader, val_loader, test_loader, info

    @staticmethod
    def _make_loader(
        X: np.ndarray,
        Z: Optional[np.ndarray],
        Y: np.ndarray,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoader:
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(Y, dtype=torch.float32)
        if Z is not None:
            z_tensor = torch.tensor(Z, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, z_tensor, y_tensor)
        else:
            dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
