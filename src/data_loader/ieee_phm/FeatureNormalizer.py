import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict

class FeatureNormalizer:
    """
    Handles feature normalization and denormalization.
    Fits on training data only, transforms train/val/test.
    Also handles target (RUL) normalization.
    """
    def __init__(self, scaler_type: str = 'minmax'):
        self.scaler_type = scaler_type
        if scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        # For target: always MinMax to [0, 1]
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit scalers on training data.
        X_train: (n_samples, seq_len, n_features) or (n_total_points, n_features)
        y_train: (n_samples,)
        """
        # Reshape X to 2D for fitting
        if X_train.ndim == 3:
            n_samples, seq_len, n_features = X_train.shape
            X_flat = X_train.reshape(-1, n_features)
        else:
            X_flat = X_train
        
        self.feature_scaler.fit(X_flat)
        self.target_scaler.fit(y_train.reshape(-1, 1))
        self.is_fitted = True
        
        print(f"Normalizer fitted ({self.scaler_type}):")
        print(f"  Feature shape used for fitting: {X_flat.shape}")
        print(f"  Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        return self
    
    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform features. X: (n_samples, seq_len, n_features)"""
        assert self.is_fitted, "Normalizer not fitted yet!"
        if X.ndim == 3:
            n, s, f = X.shape
            X_flat = X.reshape(-1, f)
            X_norm = self.feature_scaler.transform(X_flat)
            return X_norm.reshape(n, s, f).astype(np.float32)
        else:
            return self.feature_scaler.transform(X).astype(np.float32)
    
    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Transform target. y: (n_samples,)"""
        assert self.is_fitted, "Normalizer not fitted yet!"
        return self.target_scaler.transform(y.reshape(-1, 1)).flatten().astype(np.float32)
    
    def inverse_transform_X(self, X_norm: np.ndarray) -> np.ndarray:
        """Denormalize features."""
        if X_norm.ndim == 3:
            n, s, f = X_norm.shape
            X_flat = X_norm.reshape(-1, f)
            X_orig = self.feature_scaler.inverse_transform(X_flat)
            return X_orig.reshape(n, s, f).astype(np.float32)
        else:
            return self.feature_scaler.inverse_transform(X_norm).astype(np.float32)
    
    def inverse_transform_y(self, y_norm: np.ndarray) -> np.ndarray:
        """Denormalize target (RUL)."""
        return self.target_scaler.inverse_transform(
            y_norm.reshape(-1, 1)
        ).flatten().astype(np.float32)
    
    def get_params(self) -> Dict:
        """Return scaler parameters for saving/loading."""
        params = {'scaler_type': self.scaler_type, 'is_fitted': self.is_fitted}
        if self.scaler_type == 'minmax':
            params['feature_min'] = self.feature_scaler.data_min_.tolist()
            params['feature_max'] = self.feature_scaler.data_max_.tolist()
            params['target_min'] = float(self.target_scaler.data_min_[0])
            params['target_max'] = float(self.target_scaler.data_max_[0])
        elif self.scaler_type == 'standard':
            params['feature_mean'] = self.feature_scaler.mean_.tolist()
            params['feature_std'] = self.feature_scaler.scale_.tolist()
            params['target_min'] = float(self.target_scaler.data_min_[0])
            params['target_max'] = float(self.target_scaler.data_max_[0])
        return params

